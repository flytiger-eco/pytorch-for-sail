"""Gluon codegen for 2D single-loop sum; math expressions and dtypes follow SIMD lowering."""
from __future__ import annotations

import ast
import copy
import dataclasses

from ..runtime.preserved_reduction import UnsupportedReductionContract


@dataclasses.dataclass(frozen=True)
class GluonKernelSource:
    source: str
    argument_names: tuple[str, ...]
    layout_argument: str
    function_name: str = ""


class GluonCodegen(ast.NodeTransformer):
    """Convert a SIMD function with completed index/dtype lowering to the explicit-layout DSL.

    Function structure and the operator whitelist are the admission criteria. Only Python
    AST is processed here; TTGIR is never rewritten. The real layout of the standalone
    reduction is passed in after its first autotune and must not be guessed from launch configs.
    """

    layout_argument = "INDUCTOR_REDUCTION_LAYOUT"
    _calls = frozenset(
        {
            "program_id", "load", "store", "where", "sum", "exp", "exp2",
            "full", "arange", "broadcast_to", "sigmoid",
        }
    )
    _types = frozenset(
        {"constexpr", "float16", "bfloat16", "float32", "float64", "int1", "int32", "int64"}
    )
    _binary_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.BitAnd,
        ast.BitOr,
    )
    _unary_ops = (ast.UAdd, ast.USub, ast.Invert)
    _compare_ops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    @classmethod
    def generate(cls, source: str, *, function_name: str | None = None) -> GluonKernelSource:
        module = ast.parse(source)
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        if function_name is not None:
            functions = [node for node in functions if node.name == function_name]
        if len(functions) != 1:
            raise UnsupportedReductionContract("exactly one SIMD kernel function is required")
        function = copy.deepcopy(functions[0])
        arguments = tuple(arg.arg for arg in function.args.args)
        if (
            cls.layout_argument in arguments
            or function.args.posonlyargs
            or function.args.kwonlyargs
            or function.args.defaults
            or function.args.vararg
            or function.args.kwarg
            or not {"XBLOCK", "R0_BLOCK", "r0_numel"}.issubset(arguments)
        ):
            raise UnsupportedReductionContract("unsupported SIMD kernel argument contract")
        loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
        if len(loops) != 1:
            raise UnsupportedReductionContract("only single-loop reductions are supported, not persistent or multi-loop")
        loop = loops[0]
        expected_loop = ast.parse("range(0, r0_numel, R0_BLOCK)", mode="eval").body
        if (
            not isinstance(loop.target, ast.Name)
            or loop.orelse
            or ast.dump(loop.iter) != ast.dump(expected_loop)
        ):
            raise UnsupportedReductionContract("reduction loop bounds or stride do not match")
        for node in ast.walk(function):
            if isinstance(
                node,
                (
                    ast.If,
                    ast.IfExp,
                    ast.While,
                    ast.With,
                    ast.Try,
                    ast.Raise,
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                    ast.Await,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.Delete,
                    ast.Global,
                    ast.Nonlocal,
                    ast.NamedExpr,
                ),
            ):
                raise UnsupportedReductionContract("unsupported control flow or side effect")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if not all(isinstance(target, ast.Name) for target in targets):
                    raise UnsupportedReductionContract("non-local assignment is not supported")
            if isinstance(node, ast.BinOp) and not isinstance(node.op, cls._binary_ops):
                raise UnsupportedReductionContract(
                    f"unsupported binary operator: {type(node.op).__name__}"
                )
            if isinstance(node, ast.UnaryOp) and not isinstance(node.op, cls._unary_ops):
                raise UnsupportedReductionContract(
                    f"unsupported unary operator: {type(node.op).__name__}"
                )
            if isinstance(node, ast.BoolOp):
                raise UnsupportedReductionContract(
                    f"unsupported boolean operator: {type(node.op).__name__}"
                )
            if isinstance(node, ast.Compare) and (
                len(node.ops) != 1
                or not isinstance(node.ops[0], cls._compare_ops)
            ):
                raise UnsupportedReductionContract("unsupported comparison operator")
        sums = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == "sum"
        ]
        if len(sums) != 1 or any(sum_node in list(ast.walk(loop)) for sum_node in sums):
            raise UnsupportedReductionContract("only a single sum after the loop is supported")
        if (
            len(sums[0].args) != 2
            or not isinstance(sums[0].args[1], ast.Constant)
            or sums[0].args[1].value != 1
            or sums[0].keywords
        ):
            raise UnsupportedReductionContract("only sum with axis=1 and no extra arguments is supported")
        emitter = cls()
        function.decorator_list = []
        function = emitter.visit(function)
        function.args.args.append(
            ast.arg(arg=cls.layout_argument, annotation=ast.Attribute(
                value=ast.Name(id="gl", ctx=ast.Load()), attr="constexpr", ctx=ast.Load()
            ))
        )
        function.decorator_list = [ast.Attribute(
            value=ast.Name(id="gluon", ctx=ast.Load()), attr="jit", ctx=ast.Load()
        )]
        ast.fix_missing_locations(function)
        text = (
            "from triton.experimental import gluon\n"
            "from triton.experimental.gluon import language as gl\n"
            "from triton.experimental.gluon.language import SliceLayout\n\n"
            + ast.unparse(function) + "\n"
        )
        return GluonKernelSource(text, arguments, cls.layout_argument, function.name)

    def _layout(self) -> ast.Name:
        return ast.Name(id=self.layout_argument, ctx=ast.Load())

    @staticmethod
    def _call(name: str, args: list[ast.expr], keywords: list[ast.keyword] | None = None) -> ast.Call:
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="gl", ctx=ast.Load()), attr=name, ctx=ast.Load()),
            args=args,
            keywords=keywords or [],
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == "tl":
            if node.attr not in self._calls | self._types:
                raise UnsupportedReductionContract(f"unsupported Gluon operator or dtype: tl.{node.attr}")
            node.value.id = "gl"
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name):
            if node.func.id not in {"range", "float"}:
                raise UnsupportedReductionContract(f"unsupported function: {node.func.id}")
            return self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            raise UnsupportedReductionContract("indirect function calls are not supported")
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "tl"):
            if node.func.attr == "to" and not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"libdevice", "tl_math", "triton_helpers"}
            ):
                return self.generic_visit(node)
            raise UnsupportedReductionContract(f"unsupported function: {ast.unparse(node.func)}")
        name = node.func.attr
        if name not in self._calls:
            raise UnsupportedReductionContract(f"unsupported Gluon function: tl.{name}")
        args = [self.visit(arg) for arg in node.args]
        keywords = [self.visit(keyword) for keyword in node.keywords]
        if name == "arange":
            if (
                len(args) != 2 or keywords
                or not isinstance(args[0], ast.Constant) or args[0].value != 0
                or not isinstance(args[1], ast.Name)
                or args[1].id not in {"XBLOCK", "R0_BLOCK"}
            ):
                raise UnsupportedReductionContract("unsupported arange iteration domain")
            dim = 1 if args[1].id == "XBLOCK" else 0
            layout = ast.Call(
                func=ast.Name(id="SliceLayout", ctx=ast.Load()),
                args=[ast.Constant(value=dim), self._layout()], keywords=[],
            )
            keywords.append(ast.keyword(arg="layout", value=layout))
        elif name == "full":
            if len(args) != 3 or keywords or not self._two_dimensional(args[0]):
                raise UnsupportedReductionContract("only explicit 2D full is supported")
            keywords.append(ast.keyword(arg="layout", value=self._layout()))
        elif name == "broadcast_to":
            if len(args) != 2 or keywords or not self._two_dimensional(args[1]):
                raise UnsupportedReductionContract("only explicit 2D broadcast is supported")
            template = self._call(
                "full", [args[1], ast.Constant(value=0), ast.Attribute(
                    value=ast.Name(id="gl", ctx=ast.Load()), attr="int1", ctx=ast.Load()
                )], [ast.keyword(arg="layout", value=self._layout())],
            )
            return ast.Subscript(value=self._call("broadcast", [args[0], template]), slice=ast.Constant(value=0), ctx=ast.Load())
        elif name == "sigmoid":
            if len(args) != 1 or keywords:
                raise UnsupportedReductionContract("unsupported sigmoid arguments")
            # Identical to the original formula in Triton language.standard.sigmoid.
            return ast.BinOp(
                left=ast.Constant(value=1), op=ast.Div(),
                right=ast.BinOp(left=ast.Constant(value=1), op=ast.Add(),
                    right=self._call("exp", [ast.UnaryOp(op=ast.USub(), operand=args[0])])),
            )
        return self._call(name, args, keywords)

    @staticmethod
    def _two_dimensional(node: ast.expr) -> bool:
        if not isinstance(node, (ast.List, ast.Tuple)) or len(node.elts) != 2:
            return False
        return all(
            (isinstance(dim, ast.Name) and dim.id in {"XBLOCK", "R0_BLOCK", "RBLOCK"})
            or (isinstance(dim, ast.Constant) and dim.value == 1)
            for dim in node.elts
        )
