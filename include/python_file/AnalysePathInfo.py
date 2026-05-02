from typing import List, Tuple


def parse_path_info(path_info_file: str) -> Tuple[List[str], List[str], List[str]]:
    """
    按块读取 path_info.txt，兼容两种格式：

    Return:
      (base_paths, query_paths)
    """
    base_paths: List[str] = []
    query_paths: List[str] = []

    block: List[str] = []

    def _flush_block():
        nonlocal block

        if not block:
            return

        # 至少需要 base 和 query 两行
        if len(block) < 2:
            raise ValueError(
                f"Invalid block with {len(block)} non-empty lines: {block}. "
                "Each block must contain at least 2 lines: base, query."
            )
        # 兼容旧格式：如果有第三行 graph_path，直接忽略
        if len(block) > 3:
            raise ValueError(
                f"Invalid block with {len(block)} non-empty lines: {block}. "
                "Each block should be either 2 lines: base, query, "
                "or 3 lines: base, query, graph."
            )
        base_paths.append(block[0])
        query_paths.append(block[1])
        block = []

    with open(path_info_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if line == "":
                _flush_block()
            else:
                block.append(line)
    _flush_block()

    return base_paths, query_paths


def load_base_paths(path_info_file: str) -> List[str]:
    base_paths, _ = parse_path_info(path_info_file)
    return base_paths


def load_query_paths(path_info_file: str) -> List[str]:
    _, query_paths = parse_path_info(path_info_file)
    return query_paths