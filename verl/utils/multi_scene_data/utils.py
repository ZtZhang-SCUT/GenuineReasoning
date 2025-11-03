import json
import re
import logging
logger = logging.getLogger(__file__)


# def _parse_json(text: str, debug: bool = False):
#     """
#     尝试安全解析 JSON 文本。容忍 Markdown ```json``` 包裹，并自动：
#       1) 修复非法的反斜杠转义（如 \$、\[ 等，但不改变已多重转义的情况）；
#       2) 在字符串字面量内转义实际控制字符（换行、回车、制表符及其它 <0x20 的字符）；
#     按顺序尝试：直接解析 -> 修复非法转义 -> 修复字符串内控制字符 -> 再次解析。
#     返回解析后的 Python 对象或 None（并在 debug 时打印错误）。
#     """

#     def strip_markdown(s: str) -> str:
#         s = s.strip()
#         if s.startswith("```json"):
#             # 只移除最外层的 ```json ... ```
#             parts = s.split("```json", 1)[1].split("```", 1)
#             return parts[0].strip() if parts else s
#         return s

#     def fix_invalid_backslashes(s: str) -> str:
#         # """
#         # 将那些真正非法（即 \ 后不是合法 JSON 转义且该 '\' 没有被前面反斜杠转义）的单个反斜杠 '补为' 双反斜杠。
#         # 保持合法转义（\" \\ \/ \b \f \n \r \t \uXXXX）不变；已多层转义也不变。
#         # """
#         def repl(match):
#             prefix = match.group(1)  # 连续的反斜杠
#             bad_char = match.group(2)
#             bs_count = len(prefix)
#             # 如果 prefix 个数是偶数，说明最后一个反斜杠是被转义的（即不需要我们处理）
#             # 例如: '\\$' -> prefix is '\\' (2) -> even -> 不处理
#             if bs_count % 2 == 0:
#                 return prefix + bad_char
#             # 否则是奇数，说明末尾的反斜杠“实际”起作用 -> 这是非法转义，需加倍
#             # 把 ...\x 变成 ...\\x
#             return prefix + "\\" + bad_char

#         # 匹配连续若干反斜杠后跟随的下一个字符，该字符不是合法 JSON 转义起始字符
#         # 注意使用 DOTALL 不是必须；这里只需要匹配普通字符（包括换行）
#         pattern = r'(\\+)(?!["\\/bfnrtu])(.?)'
#         return re.sub(pattern, repl, s)

#     def escape_control_chars_in_strings(s: str) -> str:
#         # """
#         # 扫描 JSON 文本，把所有字符串字面量内部的实际控制字符（newline, tab, carriage）
#         # 转为对应的转义序列。保留已有的转义序列不变。
#         # 实现方法：状态机遍历，识别是否在字符串内部；处理转义字符 '\\'，并且对真正的控制字符替换为 \\n, \\r, \\t 或 \\u00XX。
#         # """
#         out = []
#         i = 0
#         n = len(s)
#         in_string = False
#         while i < n:
#             ch = s[i]
#             if ch == '"':
#                 # 判断这个 " 是否被转义（看其前面的连续反斜杠数量）
#                 # 如果前面有奇数个反斜杠则被转义，不作为字符串界定符
#                 j = i - 1
#                 bs = 0
#                 while j >= 0 and s[j] == '\\':
#                     bs += 1
#                     j -= 1
#                 if bs % 2 == 0:
#                     # 未被转义 -> 切换字符串状态
#                     in_string = not in_string
#                 out.append(ch)
#                 i += 1
#                 continue

#             if not in_string:
#                 out.append(ch)
#                 i += 1
#                 continue

#             # 在字符串内部
#             if ch == '\\':
#                 # 保留反斜杠及其后一字符（如果存在），以免破坏合法的转义序列
#                 if i + 1 < n:
#                     out.append(ch)
#                     out.append(s[i + 1])
#                     i += 2
#                 else:
#                     out.append(ch)
#                     i += 1
#                 continue

#             # 如果是实际控制字符（ASCII < 0x20），需要转义
#             code = ord(ch)
#             if code == 0x0A:          # \n
#                 out.append('\\n')
#             elif code == 0x0D:        # \r
#                 out.append('\\r')
#             elif code == 0x09:        # \t
#                 out.append('\\t')
#             elif code < 0x20:
#                 # 其它控制字符，用 unicode 转义
#                 out.append('\\u%04x' % code)
#             else:
#                 out.append(ch)
#             i += 1

#         return ''.join(out)

#     # 主流程
#     text = strip_markdown(text)

#     # 1) 直接尝试解析
#     try:
#         # return json.loads(text, strict=False)
#         return json.loads(text)
#     except json.JSONDecodeError as e1:
#         if debug:
#             print("第一次解析失败:", e1)

#     # 2) 尝试修复非法转义（Invalid \escape）
#     fixed1 = fix_invalid_backslashes(text)
#     try:
#         # return json.loads(fixed1, strict=False)
#         return json.loads(fixed1)
#     except json.JSONDecodeError as e2:
#         if debug:
#             print("修复非法转义后解析失败:", e2)

#     # 3) 如果仍失败，尝试将字符串内部的控制字符转义（例如实际换行）
#     fixed2 = escape_control_chars_in_strings(fixed1)
#     if debug:
#         print("----- 经过修复（非法转义 + 字符串控制字符）后的片段 -----")
#         print(fixed2[:800])
#     try:
#         # return json.loads(fixed2, strict=False)
#         return json.loads(fixed2)
#     except json.JSONDecodeError as e3:
#         if debug:
#             print("最终解析仍失败:", e3)
#         return None


def _parse_json(text: str, debug: bool = False):
    """
    更鲁棒的 JSON 解析器：
    - 移除 ```json ... ``` 包裹；
    - 在字符串字面量内部：
        * 将那些会被 JSON 误解析为控制字符的单斜杠 + 多字母序列（如 \\frac）改为双斜杠 (\\\\frac)；
        * 保留合法 JSON 转义（\\n, \\t, \\\", \\\\, \\/, \\b, \\f, \\r, \\uXXXX）；
        * 如果遇到真实控制字符（换行、回车、制表和其它 < 0x20）则转义为 \\n / \\r / \\t / \\uXXXX；
    - 尽量只对“活跃”的（前面有奇数个斜杠）情况进行加倍；若前导反斜杠数为偶数（例如已经是 \\\\frac），则不改。
    """
    
    def strip_markdown(s: str) -> str:
        s = s.strip()
        if s.startswith("```json"):
            parts = s.split("```json", 1)[1].split("```", 1)
            return parts[0].strip() if parts else s
        return s

    text = strip_markdown(text)

    # 快速尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if debug:
            print("第一次解析失败:", e)

    # 我们将逐字符扫描，专注于字符串字面量内部做精准修复
    out_chars = []
    i = 0
    n = len(text)
    in_string = False

    while i < n:
        ch = text[i]

        # 处理双引号是否作为字符串边界（非被转义的 "）
        if ch == '"':
            # 计算前导反斜杠数量判断是否被转义
            j = i - 1
            bs = 0
            while j >= 0 and text[j] == '\\':
                bs += 1
                j -= 1
            if bs % 2 == 0:
                # 未被转义的双引号 => 切换字符串状态
                in_string = not in_string
            out_chars.append(ch)
            i += 1
            continue

        if not in_string:
            out_chars.append(ch)
            i += 1
            continue

        # ---------- 现在我们在字符串内部 ----------
        # 若遇到反斜杠，处理可能的转义或 LaTeX-like 命令
        if ch == '\\':
            # 统计连续反斜杠长度
            j = i
            while j < n and text[j] == '\\':
                j += 1
            bs_count = j - i  # 连续反斜杠的数量
            # 看看后面紧邻的字符（如果有）
            k = j
            next_char = text[k] if k < n else ''
            # 如果后面是 'u' 并且后续4个是 hex，则这是 \uXXXX，保留原样（拷贝这些反斜杠和 u 与后面的 4 字符）
            if next_char == 'u' and k + 5 < n and re.match(r'[0-9a-fA-F]{4}', text[k+1:k+5]):
                # 把所有这些字符原样复制
                out_chars.append('\\' * bs_count)
                out_chars.append('u')
                out_chars.append(text[k+1:k+5])
                i = k + 5
                continue

            # 如果后面是合法单字符 JSON 转义之一（" \/ b f n r t）
            if next_char in '"\\/bfnrt':
                # 但要注意：像 \f 可能是 LaTeX \frac 的开始（后面还有字母）
                # 检查 next_char 是否字母，且后面还有字母（构成多字母命令），
                # 如果是多字母命令（例如 \frac, \textbf），则我们应该把该“活跃”的反斜杠加倍，
                # 否则保留为合法转义。
                is_letter_cmd = False
                if next_char.isalpha():
                    # 查看 next_char 后面是否还有字母（构成多字母序列）
                    look = k + 1
                    while look < n and text[look].isalpha():
                        look += 1
                    if look - (k + 1) >= 1:
                        # 后面至少还有一个字母 -> 多字母命令
                        is_letter_cmd = True

                if is_letter_cmd:
                    # 如果前导反斜杠数是奇数（活跃），就把最后一个活跃的斜杠加倍
                    if bs_count % 2 == 1:
                        out_chars.append('\\' * (bs_count + 1))
                    else:
                        out_chars.append('\\' * bs_count)
                    # 不消耗后面的字母，这里只复制接下来字符（不会特殊处理）
                    i = j
                    continue
                else:
                    # 合法短转义，直接拷贝（不做额外处理）
                    out_chars.append('\\' * bs_count)
                    # copy the next char as well (part of the escape)
                    if k < n:
                        out_chars.append(text[k])
                        i = k + 1
                    else:
                        i = j
                    continue

            # 如果 next_char 是字母（比如 f, t, r ...）并且后面还有字母 -> 很可能是 LaTeX 命令
            if next_char.isalpha():
                # 取出连续的字母序列长度
                look = k
                while look < n and text[look].isalpha():
                    look += 1
                letters_len = look - k
                if letters_len >= 2:
                    # 多字母命令（如 frac, textbf）：确保活跃的单斜杠被转为双斜杠
                    if bs_count % 2 == 1:
                        out_chars.append('\\' * (bs_count + 1))
                    else:
                        out_chars.append('\\' * bs_count)
                    # 不改变字母本身，继续到字母开始位置（i -> j）
                    i = j
                    continue
                else:
                    # 单字母后面不是字母 -> treat as normal (可能是 \f alone)
                    out_chars.append('\\' * bs_count)
                    i = j
                    continue

            # 其它情况（后面不是字母也不是合法短转义的字符，例如 \$、\[、\{ 等）
            # 如果前导反斜杠是奇数（说明当前斜杠是真的要转义下一个字符），我们需要把它加倍
            if bs_count % 2 == 1:
                out_chars.append('\\' * (bs_count + 1))
            else:
                out_chars.append('\\' * bs_count)
            i = j
            continue

        # 如果不是反斜杠，在字符串内部遇到控制字符（比如真实换行、制表等）需要转义
        code = ord(ch)
        if code == 0x0A:
            out_chars.append('\\n')
        elif code == 0x0D:
            out_chars.append('\\r')
        elif code == 0x09:
            out_chars.append('\\t')
        elif code < 0x20:
            out_chars.append('\\u%04x' % code)
        else:
            out_chars.append(ch)
        i += 1

    fixed_text = ''.join(out_chars)

    if debug:
        print("----- 修复后片段（前800字符） -----")
        print(fixed_text[:800])

    # 最后尝试解析（先尝试 strict=False）
    try:
        return json.loads(fixed_text)
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        if debug:
            print("最终解析失败:", e)
        # return None
        raise(e)