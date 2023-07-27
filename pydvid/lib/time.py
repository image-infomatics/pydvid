
import contextlib
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_TIMESTAMP = datetime.strptime('2018-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

@contextlib.contextmanager
def Timer(msg=None, logger=None, level=logging.INFO, log_start=True):
    """
    Simple context manager that acts as a wall-clock timer.

    Args:
        msg:
            Optional message to be logged at the start
            and stop of the timed period.
        logger:
            Which logger to write the message to.

    Example:
        >>> with Timer("Doing stuff") as timer:
        ...     # do stuff here
        >>>
        >>> print(timer.seconds)
        >>> print(timer.timedelta)
    """
    result = _TimerResult()
    if msg:
        logger = logger or logging.getLogger(__name__)
        if log_start:
            logger.log(level, msg + '...')
    try:
        yield result
    except BaseException as ex:
        result.stop = time.time()
        if msg:
            logger.error(msg + f' failed due to {type(ex).__name__} after {result.timedelta}')
        raise
    else:
        result.stop = time.time()
        if msg:
            logger.log(level, msg + f' took {result.timedelta}')


class _TimerResult(object):
    """
    Helper class, yielded by the Timer context manager.
    """
    def __init__(self):
        self.start = time.time()
        self.stop = None

    @property
    def seconds(self):
        if self.stop is None:
            return time.time() - self.start
        else:
            return self.stop - self.start

    @property
    def timedelta(self):
        return timedelta(seconds=self.seconds)

def parse_timestamp(ts, default=DEFAULT_TIMESTAMP, default_timezone="US/Eastern"):
    """
    Parse the given timestamp as a datetime object.
    If it is already a datetime object, it will be returned as-is.
    If it is None, then the given default timestamp will be returned.

    If the timestamp is not yet "localized", it will be assigned a
    timezone according to the default_timezone argument.
    (That is, we assume the time in the string was recorded in the specified timezone.)
    Localized timestamps include a suffix to indicate the offset from UTC.
    See the examples below.

    Note:
        By POSIX timestamp conventions, the +/- sign of the timezone
        offset might be reversed of what you expected unless you're
        already familiar with this sort of thing.

    Example timestamps:

        2018-01-01             (date only)
        2018-01-01 00:00       (date and time)
        2018-01-01 00:00:00    (date and time with seconds)
        2018-01-01 00:00:00.0  (date and time with microseconds)

        2018-01-01 00:00-4:00  (date and time, localized with some US timezone offset)

    Returns:
        datetime

    """
    if ts is None:
        ts = copy.copy(default)

    if isinstance(ts, (datetime, pd.Timestamp)):
        return ts

    if isinstance(ts, str):
        ts = pd.Timestamp(ts)

    if ts.tzinfo is None and default_timezone is not None:
        ts = pd.Timestamp.tz_localize(ts, pytz.timezone(default_timezone))

    return ts