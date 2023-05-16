r"""Interface for hyperparameter declaration and optimization.

Each experiment run (see
:py:class:`~dair_pll.experiment.SupervisedLearningExperiment`\ ) can
have its
hyperparameters optimized via :py:mod:`optuna`\ . By design, each experiment
is fully described via a
:py:class:`~dair_pll.experiment.SupervisedLearningExperimentConfig` object.
This file implements a :py:class:`Hyperparameter` class, which can be declared
as a member variable of a
:py:class:`~dair_pll.experiment.SupervisedLearningExperimentConfig`,
or recursively as one of its own :py:func:`~dataclasses.dataclass` members.

The following hyperparameters types and priors are supported:

    * :py:class:`Scalar`, a :py:class:`Float` or :py:class:`Int` which is
      either uniformly or log-uniformly distributed.
    * :py:class:`Categorical`, a list of :py:class:`float`\ , :py:class:`int`\
      , or :py:class:`str` types which are selected from uniformly.

"""
from abc import abstractmethod, ABC
from dataclasses import is_dataclass
from typing import Tuple, TypeVar, Sequence, List, Union, Optional, \
    Dict, Generic, Callable, Any

from optuna.trial import Trial

ValueType = Union[int, float, str]
ValueDict = Dict[str, ValueType]

ScalarT = TypeVar('ScalarT', int, float)
r"""Templating type hint for :py:class:`Scalar`\ s."""


class Hyperparameter(ABC):
    """Class for declaring and sampling hyperparameters.

    Hyperparameters have both a :py:attr:`value` and a
    :py:attr:`distribution` from which vales might be selected.

    Declaration of a hyperparameter in a configuration may look like::

        @dataclass
        class XXXConfig:
            int_par: Int = Int(5, (0, 10))
            float_par: Float = Float(0.1, (1e-4, 1e3), log=True)
            cat_par: Categorical = Categorical('val2', ['val0','val1', 'val2'])

    In these cases, the first argument is the default :py:attr:`value` of the
    hyperparameter. However, at hyperparameter optimization time,
    :py:mod:`optuna` will select hyperparameters from the
    :py:attr:`distribution` via the :py:meth:`suggest` function. Some
    hyperparameter types have a default distribution as described in their
    documentation."""

    value: ValueType
    """Hyperparameter's current value."""
    distribution: Sequence[ValueType]
    """Parameters for distribution from which to select value."""

    def __init__(self, value: ValueType, distribution: Sequence[ValueType]):
        self.distribution = distribution
        self.value = value

    def set(self, value: ValueType):
        """Setter for underlying hyperparameter value."""
        self.value = value

    def __repr__(self) -> str:
        """Human-readable representation of underlying hyperparameter value."""
        return f'{type(self).__qualname__}({str(self.value)})'

    @abstractmethod
    def suggest(self, trial: Trial, name: str) -> ValueType:
        r"""Suggests a value for the hyperparameter.

        This function is abstract as to facilitate specialization for
        inheriting types.

        Args:
            trial: Optuna trial in which parameter is being suggested.
            name: Name of hyperparameter.

        Returns:
            Suggested hyperparameter value.
        """


class Scalar(Hyperparameter, ABC, Generic[ScalarT]):
    r"""Abstract scalar hyperparameter type.

    Defines a uniform or log-uniform distribution over a scalar type, such as
    integers or real numbers.

    The bounds of the distribution can either be specified as a tuple in the
    :py:attr:`distribution` attribute, or set as a default based on the
    provided :py:attr:`value` in the abstract method :py:meth:`default_range`\ .
    """
    value: ScalarT
    """Scalar value of hyperparameter."""
    distribution: Tuple[ScalarT, ScalarT]
    """Bounds of scalar distribution in format (lower, upper)."""
    log: bool
    """Whether the distribution is uniform or log-uniform."""

    def __init__(self,
                 value: ScalarT,
                 distribution: Optional[Tuple[ScalarT, ScalarT]] = None,
                 log: bool = False):
        if not distribution:
            distribution = self.default_range(value, log)
        assert distribution[1] >= distribution[0]
        if log:
            assert distribution[0] > 0

        super().__init__(value, distribution)
        self.log = log

    @abstractmethod
    def default_range(self, value: ScalarT,
                      log: bool) -> Tuple[ScalarT, ScalarT]:
        """Returns default range for Scalar, depending on provided value."""


INT_LOG_WIDTH = 2**3
INT_ABS_WIDTH = 2


class Int(Scalar):
    """Integer scalar hyperparameter."""
    value: int
    distribution: Tuple[int, int]

    def default_range(self, value: int, log: bool) -> Tuple[int, int]:
        """Default bounds for integer hyperparameter.
        
        Returns ``(max(1, value // RANGE), value * RANGE)``, where ``RANGE``
        is ``8`` in the log-uniform case and ``2`` otherwise.
        
        Args:
            value: Default value of hyperparameter.
            log: Whether the distribution is uniform or log-uniform.

        Returns:
            Default lower/upper bounds.
        """
        width = INT_LOG_WIDTH if log else INT_ABS_WIDTH
        return max(1, value // width), value * width

    def suggest(self, trial: Trial, name: str) -> int:
        r"""Returns suggested (log)-uniform distributed integer."""
        return trial.suggest_int(name, *self.distribution, log=self.log)


FLOAT_LOG_WIDTH = 1e2
FLOAT_ABS_WIDTH = 2.


class Float(Scalar):
    """Real number (floating-point) scalar hyperparameter."""
    value: float
    distribution: Tuple[float, float]

    def default_range(self, value: float, log: bool) -> Tuple[float, float]:
        """Default bounds for float hyperparameter.

        Returns ``(value / RANGE, value * RANGE)``, where ``RANGE``
        is ``100`` in the log-uniform case and ``2`` otherwise.

        Args:
            value: Default value of hyperparameter.
            log: Whether the distribution is uniform or log-uniform.

        Returns:
            Default lower/upper bounds.
        """
        width = FLOAT_LOG_WIDTH if log else FLOAT_ABS_WIDTH
        return value / width, value * width

    def suggest(self, trial: Trial, name: str) -> float:
        r"""Returns suggested (log)-uniform distributed float."""
        return trial.suggest_float(name, *self.distribution, log=self.log)


# Only one new public method, but just happens to be a particularly simple
# inheriting type of Hyperparameter.
class Categorical(Hyperparameter):  # pylint: disable=too-few-public-methods
    """Categorical hyperparameter."""
    value: ValueType
    distribution: List[ValueType]

    def suggest(self, trial: Trial, name: str) -> ValueType:
        """Suggests from listed values in distribution uniformly."""
        suggestion = trial.suggest_categorical(name, self.distribution)
        assert isinstance(suggestion, (float, int, str))
        return suggestion


def is_dataclass_instance(obj) -> bool:
    r"""Helper function to check if input object is a
    :py:func:`~dataclasses.dataclass`\ ."""
    return is_dataclass(obj) and not isinstance(obj, type)


def traverse_config(config: Any,
                    callback: Callable[[str, Hyperparameter], None],
                    namespace: str = '') -> None:
    r"""Recursively searches through ``config`` and its member
    :py:func:`~dataclasses.dataclass`\ es, for member
    :py:class:`Hyperparameter` objects.

    While traversing the tree, maintains a `namespace` constructed from a
    concatenation of the attributes' names.

    When a :py:class:`Hyperparameter` 'h' under attribute name `attr` is
    encountered, this function calls :py:arg:`callback` with inputs
    ``(namespace + attr, h)``\ .

    Args:
        config: Configuration :py:func:`~dataclasses.dataclass` \ .
        callback: Callback performed on each :py:class:`Hyperparameter`\ .
        namespace: (Optional/internal) prefix for naming hyperparameters.
    """
    assert is_dataclass_instance(config)

    for field in config.__dataclass_fields__:
        value = getattr(config, field)
        if is_dataclass_instance(value):
            subspace = f'{namespace}{field}.'
            traverse_config(value, callback, subspace)
        if isinstance(value, Hyperparameter):
            name = namespace + field
            callback(name, value)


def generate_suggestion(config, trial: Trial) -> ValueDict:
    r"""Suggests a value all hyperparameters in configuration (but does not
    set these values).

    Recursively searches through ``config`` and its member
    :py:func:`~dataclasses.dataclass`\ es, and generates a suggestion for
    each contained :py:class:`Hyperparameter`\ .

    Args:
        config: Configuration :py:func:`~dataclasses.dataclass` \ .
        trial: Optuna trial in which parameters are being suggested.

    Returns:
        Suggested hyperparameter value dictionary.
    """
    assert is_dataclass_instance(config)

    out_dict = {}

    def callback(name: str, hyperparameter: Hyperparameter):
        out_dict[name] = hyperparameter.suggest(trial, name)

    traverse_config(config, callback)

    return out_dict


def load_suggestion(config: Any, suggestion: ValueDict) -> None:
    r"""Fill all hyperparameters in configuration with suggestions.

    Recursively searches through ``config`` and its member
    :py:func:`~dataclasses.dataclass`\ es, and sets the values to the
    suggestion for each contained :py:class:`Hyperparameter`\ .

    The ``suggestion`` is assumed to be generated by running
    :py:func:`generate_suggestion` on an identical type to ``config``.

    Args:
        config: Configuration :py:func:`~dataclasses.dataclass` \ .
        suggestion: Suggested hyperparameter set.
    """
    assert is_dataclass_instance(config)

    def callback(name: str, hyperparameter: Hyperparameter):
        hyperparameter.set(suggestion[name])

    traverse_config(config, callback)


def hyperparameter_values(config: Any) -> ValueDict:
    r"""Lists current values for all hyperparameters in configuration.

    Recursively searches through ``config`` and its member
    :py:func:`~dataclasses.dataclass`\ es, and records value for each contained
    :py:class:`Hyperparameter`\ .

    Args:
        config: Configuration :py:func:`~dataclasses.dataclass` \ .

    Returns:
        Hyperparameter value dictionary.
    """
    assert is_dataclass_instance(config)

    out_dict = {}

    def callback(name: str, hyperparameter: Hyperparameter):
        out_dict[name] = hyperparameter.value

    traverse_config(config, callback)

    return out_dict
