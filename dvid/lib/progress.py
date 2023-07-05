import os
import sys
import inspect
import logging
import io

from tqdm import tqdm

logger = logging.getLogger(__name__)

def tqdm_proxy(iterable=None, *, logger=None, level=logging.INFO, **kwargs):
    """
    Useful as an (almost) drop-in replacement for ``tqdm`` which can be used
    in EITHER an interactive console OR a script that logs to file.

    Automatically detects whether or not sys.stdout is a file or a console,
    and configures tqdm accordingly.

    - If your code is running from an interactive console, this acts like plain ``tqdm``.
    - If your code is running from an ipython notebook, this acts like ``tqdm.notebook.tqdm``.
    - If your code is running from a batch script (i.e. printing to a log file, not the console),
      this code uses the supplied logger to periodically output a textual progress bar.
      If no logger is supplied, a logger is automatically created using the name of
      the calling module.

    Example:

        for i in tqdm_proxy(range(1000)):
            # do some stuff

    Note for JupyterLab users:

        If you get errors in this function, you need to run the following commands:

            conda install -c conda-forge ipywidgets
            jupyter nbextension enable --py widgetsnbextension
            jupyter labextension install @jupyter-widgets/jupyterlab-manager

        ...and then reload your jupyterlab session, and restart your kernel.
    """
    assert 'file' not in kwargs, \
        "There's no reason to use this function if you are providing your own output stream"

    # Special case for tqdm_proxy(range(...))
    if iterable is not None and isinstance(iterable, range) and 'total' not in kwargs:
        kwargs['total'] = (iterable.stop - iterable.start) // iterable.step

    try:
        import ipykernel.iostream
        from tqdm.notebook import tqdm as tqdm_notebook
        if isinstance(sys.stdout, ipykernel.iostream.OutStream):
            return tqdm_notebook(iterable, **kwargs)
    except ImportError:
        pass

    _tqdm = tqdm
    _file = None
    disable_monitor = False

    if not _file and os.isatty(sys.stdout.fileno()):
        _file = sys.stdout
    else:
        if logger is None:
            frame = inspect.stack()[1]
            modname = inspect.getmodulename(frame[1])
            if modname:
                logger = logging.getLogger(modname)
            else:
                logger = logging.getLogger("unknown")

        _file = TqdmToLogger(logger, level)

        # The tqdm monitor thread messes up our 'miniters' setting, so disable it.
        disable_monitor = True

        if 'ncols' not in kwargs:
            kwargs['ncols'] = 100

        if 'miniters' not in kwargs:
            # Aim for 5% updates
            if 'total' in kwargs:
                kwargs['miniters'] = kwargs['total'] // 20
            elif hasattr(iterable, '__len__'):
                kwargs['miniters'] = len(iterable) // 20

    kwargs['file'] = _file
    bar = _tqdm(iterable, **kwargs)
    if disable_monitor:
        bar.monitor_interval = 0
    return bar


class TqdmToLogger(io.StringIO):
    """
    Output stream for tqdm which will output to logger module instead of stdout.
    Copied from:
    https://github.com/tqdm/tqdm/issues/313#issuecomment-267959111
    """
    logger = None
    level = logging.INFO
    buf = ''

    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)