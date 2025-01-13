import logging
import logging.config
import sys


def setup_logger(level: int, log_file_path: str = None):
    modules = ("index", "rag", "gpt", "query")
    debug_filtr = logging.Filter()
    debug_filtr.filter = lambda record: record.levelno == logging.DEBUG and record.module in modules
    
    info_filtr = logging.Filter()
    info_filtr.filter = lambda record: record.module in modules
    
    if log_file_path:
        debug_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        info_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    else:
        debug_handler = logging.StreamHandler(sys.stdout)
        info_handler = logging.StreamHandler(sys.stdout)
    
    debug_handler.setFormatter(logging.Formatter("%(levelname)s - %(filename)s:%(lineno)d - %(message)s"))
    debug_handler.addFilter(debug_filtr)
    
    info_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    info_handler.addFilter(info_filtr)
    
    logging.basicConfig(
        level=level,
        handlers=[debug_handler, info_handler,]
    )