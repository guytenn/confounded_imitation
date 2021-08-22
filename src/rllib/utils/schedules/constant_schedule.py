from src.rllib.utils.annotations import override
from src.rllib.utils.framework import try_import_tf
from src.rllib.utils.schedules.schedule import Schedule

tf1, tf, tfv = try_import_tf()


class ConstantSchedule(Schedule):
    """
    A Schedule where the value remains constant over time.
    """

    def __init__(self, value, framework):
        """
        Args:
            value (float): The constant value to return, independently of time.
        """
        super().__init__(framework=framework)
        self._v = value

    @override(Schedule)
    def _value(self, t):
        return self._v

    @override(Schedule)
    def _tf_value_op(self, t):
        return tf.constant(self._v)
