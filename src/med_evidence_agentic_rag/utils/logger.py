"""创建项目统一的、Windows 编码安全的控制台日志。

``setup_logger`` 按模块名复用 Logger，避免热重载或多次导入产生重复 handler；可选控制台 handler
使用统一时间、级别、模块名和颜色格式。``_EncodingSafeStream`` 在系统代码页无法写入中文时退化为
Unicode 转义，确保故障日志仍可读取和追踪。
"""

import logging
import sys
from typing import TextIO

import colorlog


class _EncodingSafeStream:
    """让控制台日志在系统编码缺字时退化为可追踪的 Unicode 转义。"""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, value: str) -> int:
        encoding = self._stream.encoding
        if encoding:
            value = value.encode(encoding, errors="backslashreplace").decode(encoding)
        return self._stream.write(value)

    def flush(self) -> None:
        self._stream.flush()


def setup_logger(
    name: str,
    level: str = "DEBUG",
    console: bool = True,
) -> logging.Logger:
    """配置并返回 Logger；同名 Logger 不重复添加 handler。"""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 已配置的同名 Logger 直接复用，避免重复输出。
    if logger.handlers:
        return logger

    # 控制台只负责颜色和统一格式，日志正文由各业务模块使用中文。
    if console:
        formatter = colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "blue",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            reset=True,
        )
        handler = colorlog.StreamHandler(_EncodingSafeStream(sys.stdout))
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    setup_logger(__name__).info("日志组件测试成功")
