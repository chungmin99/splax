from typing import Callable, Type, TypeVar, TYPE_CHECKING
import jax_dataclasses as jdc
import jaxlie

if TYPE_CHECKING:
    from ._gaussian_splat import _Gaussians

T = TypeVar("T", bound="_Gaussians")


def register_gs(
    *,
    n_dim: int,
    rot_type: jdc.Static[type[jaxlie.SOBase]],
) -> Callable[[Type[T]], Type[T]]:
    """Decorator for registering Lie group dataclasses.

    Sets dimensionality class variables, and marks all methods for JIT compilation.
    """

    def _wrap(cls: Type[T]) -> Type[T]:
        # Register dimensions as class attributes.
        cls.n_dim = n_dim
        cls.rot_type = rot_type

        cls.tangent_dim = (
            cls.rot_type.tangent_dim
            + cls.n_dim  # means
            + cls.n_dim  # scale
            + 3  # colors
            + 1  # opacity
        )

        # JIT all methods.
        for f in filter(
            lambda f: not f.startswith("_")
            and callable(getattr(cls, f))
            and f != "get_batch_axes",  # Avoid returning tracers.
            dir(cls),
        ):
            setattr(cls, f, jdc.jit(getattr(cls, f)))

        return cls

    return _wrap
