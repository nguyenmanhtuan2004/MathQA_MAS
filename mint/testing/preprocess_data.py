from pydantic import BaseModel
import re
from typing import TypedDict, Optional
from openai import OpenAI


def convert_table_to_markdown(table_data):
    header = table_data[0]
    rows = table_data[1:]
    md = '| ' + ' | '.join(header) + ' |\n'
    md += '| ' + ' | '.join(['---'] * len(header)) + ' |\n'
    for row in rows:
        md += '| ' + ' | '.join(row) + ' |\n'
    return md


def tabmwp_table_to_markdown(table_for_pd):
    header = list(table_for_pd.keys())
    rows = list(zip(*[table_for_pd[h] for h in header]))
    md = '| ' + ' | '.join(header) + ' |\n'
    md += '| ' + ' | '.join(['---'] * len(header)) + ' |\n'
    for row in rows:
        md += '| ' + ' | '.join([str(cell) for cell in row]) + ' |\n'
    return md



def prepare_qa_input_with_answer_filter(data):
    results = []
    for item in data:
        # Chuyển table sang markdown nếu có
        md_table = convert_table_to_markdown(item['table']['table']) if 'table' in item and 'table' in item['table'] else ""
        # Tạo dict mapping order -> text cho paragraph
        order2text = {str(p['order']): p['text'] for p in item.get('paragraphs', [])}
        questions = []
        for q in sorted(item['questions'], key=lambda x: x['order']):
            # Bỏ các câu hỏi không cần thiết
            if q.get('answer_type') == 'multi-span':
                continue
            if q.get('answer_from') == 'text':
                continue
            # Lọc context chỉ lấy đúng các paragraph liên quan
            rel_orders = q.get('rel_paragraphs', [])
            rel_paragraphs = [order2text[o] for o in rel_orders if o in order2text]
            context = ""
            if rel_paragraphs:
                context = "\n\n".join(rel_paragraphs)
            # Nếu có bảng, nối bảng vào context
            if md_table:
                context = context + "\n\n" + md_table if context else md_table
            questions.append({
                'question': q['question'],
                'answer': q.get('answer', None),
                'context': context
            })
        # Mỗi câu hỏi là một dict riêng biệt (mỗi context riêng)
        results.extend(questions)
    return results



def normalize_answer(ans, dataset_type=None):
    # Nếu là list có 1 phần tử, lấy phần tử đó
    if isinstance(ans, list) and len(ans) == 1:
        ans = ans[0]
    # Nếu là số, kiểm tra .0
    if isinstance(ans, (int, float)):
        if isinstance(ans, float) and ans.is_integer():
            return str(int(ans))  # Chuyển về string
        return str(ans)          # Chuyển về string
    s = str(ans).replace(',', '').strip()
    # Xóa dấu $ nếu là tatqa
    if dataset_type == 'tatqa':
        s = s.replace('$', '')
    if dataset_type == "gsm8k":
        match = re.search(r'####\s*([-\d\./]+)', s)
        if match:
            num_str = match.group(1)
            # Nếu là phân số, trả luôn
            if '/' in num_str:
                return num_str
            try:
                num = float(num_str)
                if num.is_integer():
                    return str(int(num))
                return str(num)
            except ValueError:
                return num_str.strip()
    # Nếu là phân số (có dấu "/"), trả luôn
    if '/' in s:
        return s
    # Nếu là số thực hoặc số nguyên
    try:
        num = float(s)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return s


def standardize_item(item, dataset_type):
    if dataset_type == "gsm8k":
        return [{
            "question": item["question"],
            "answer": normalize_answer(item["answer"], dataset_type),
            "context": ""
        }]
    elif dataset_type == "tatqa":
        context = item.get("context", "")
        return [{
            "question": item["question"],
            "answer": normalize_answer(item["answer"], dataset_type),
            "context": context
        }]
    elif dataset_type == "tabmwp":
        context = ""
        if item.get("table_title"):
            context += item["table_title"] + "\n\n"
        if item.get("table_for_pd"):
            context += tabmwp_table_to_markdown(item["table_for_pd"])
        return [{
            "question": item["question"],
            "answer": normalize_answer(item["answer"], dataset_type),
            "context": context
        }]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
