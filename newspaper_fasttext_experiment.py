#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文维基语料上的表示学习与人民日报主题分类小实验。

运行方式：
    python -m pip install -r requirements.txt
    python newspaper_fasttext_experiment.py

默认使用：
    基本语料/train_bpe_dataset.txt
    人民日报数据/1998-01-2003版-带音.txt

脚本会在“实验输出”目录中生成分类指标 CSV、实验配置 JSON、多张图和
一份由本脚本自动生成的 DOCX 实验报告。标签由显式主题词规则产生，属于
弱标签；报告会单独说明这一评估限制。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
import sys
import unicodedata
import warnings
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable

import jieba
import numpy as np
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from gensim.models import FastText, Word2Vec
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import OptimizeWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings(
    "ignore",
    message="Unknown solver options: iprint",
    category=OptimizeWarning,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.getLogger("jieba").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WIKI = BASE_DIR / "基本语料" / "train_bpe_dataset.txt"
DEFAULT_PEOPLE = BASE_DIR / "人民日报数据" / "1998-01-2003版-带音.txt"
DEFAULT_OUTPUT = BASE_DIR / "实验输出"

LABEL_ORDER = ["经济", "国际", "体育"]
TOPIC_TERMS = {
    "经济": {
        "经济", "国民经济", "市场", "企业", "金融", "投资", "财政", "贸易",
        "出口", "进口", "银行", "股市", "股票", "工业", "农业", "外贸",
        "价格", "资金", "消费", "产值", "经营", "产品", "产业", "公司",
        "收入", "贷款", "债券", "商品", "生产", "经济增长", "经济发展",
    },
    "体育": {
        "体育", "比赛", "足球", "篮球", "排球", "乒乓球", "网球", "奥运会",
        "世界杯", "联赛", "球队", "球员", "运动员", "教练", "冠军", "金牌",
        "田径", "游泳", "运动会", "亚运会", "赛事", "赛场", "射击", "体操",
        "举重", "女足", "男足", "夺冠", "球赛", "队员",
    },
    "国际": {
        "国际", "联合国", "外交", "美国", "日本", "俄罗斯", "法国", "英国",
        "德国", "欧盟", "韩国", "朝鲜", "伊拉克", "伊朗", "中东", "海湾",
        "首相", "总统", "外长", "大使", "访问", "会谈", "谈判", "协议",
        "双边", "首脑", "国际社会", "国际组织", "欧洲", "亚洲", "非洲",
        "科索沃", "南联盟", "北约", "使馆", "外交部", "中美", "中日",
    },
}

STOPWORDS = {
    "我们", "你们", "他们", "她们", "这个", "那个", "一个", "一些", "一种",
    "可以", "已经", "进行", "通过", "因为", "所以", "但是", "并且", "其中",
    "以及", "同时", "目前", "如果", "对于", "为了", "按照", "根据", "有关",
    "自己", "没有", "不是", "什么", "怎样", "如何", "这些", "那些", "由于",
    "记者", "表示", "指出", "认为", "介绍", "有关", "工作", "问题", "方面",
}
ARTICLE_ID = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}$")
HAS_TEXT = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")


def clean_token(raw: str) -> str | None:
    """统一全角字符并清掉语料中的拼音注记、标点和停用词。"""
    token = unicodedata.normalize("NFKC", raw.split("{", 1)[0]).strip()
    token = token.strip(" \t\r\n,.;:!?，。；：！？、（）()[]【】《》“”‘’\"'…—-")
    token = token.lower()
    if len(token) < 2 or not HAS_TEXT.search(token) or token in STOPWORDS:
        return None
    return token


def reservoir_sample_wiki(path: Path, limit: int, seed: int) -> tuple[list[str], int]:
    """流式扫描文件，以蓄水池抽样保留分散在全文件中的段落。"""
    rng = random.Random(seed)
    reservoir: list[str] = []
    eligible = 0
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        for line in source:
            paragraph = line.strip()
            if len(paragraph) < 20 or len(paragraph) > 8000:
                continue
            if not re.search(r"[\u3400-\u9fff]", paragraph):
                continue
            eligible += 1
            if len(reservoir) < limit:
                reservoir.append(paragraph)
            else:
                slot = rng.randrange(eligible)
                if slot < limit:
                    reservoir[slot] = paragraph
    rng.shuffle(reservoir)
    return reservoir, eligible


def tokenize_wiki(paragraphs: Iterable[str]) -> list[list[str]]:
    sentences: list[list[str]] = []
    for paragraph in paragraphs:
        tokens = [value for item in jieba.lcut(paragraph, HMM=False) if (value := clean_token(item))]
        if len(tokens) >= 3:
            sentences.append(tokens)
    return sentences


def read_people_documents(
    path: Path, encoding: str = "gb18030"
) -> tuple[list[dict[str, Any]], int, int]:
    """将首句作为标题，其余句子聚合为正文，并按编号前三段分组。"""
    documents: OrderedDict[str, dict[str, Any]] = OrderedDict()
    sentence_count = 0
    with path.open("r", encoding=encoding, errors="replace") as source:
        for line in source:
            fields = line.strip().split()
            if len(fields) < 2 or not ARTICLE_ID.match(fields[0]):
                continue
            article_id = fields[0].rsplit("-", 1)[0]
            entry = documents.setdefault(
                article_id, {"title_tokens": None, "body_tokens": []}
            )
            tokens = [value for item in fields[1:] if (value := clean_token(item))]
            if entry["title_tokens"] is None:
                entry["title_tokens"] = tokens
            else:
                entry["body_tokens"].extend(tokens)
            sentence_count += 1
    records = []
    for article_id, entry in documents.items():
        title_tokens = entry["title_tokens"] or []
        body_tokens = entry["body_tokens"]
        # 避免只靠标题分类，也避免把标题原样作为分类特征。
        if title_tokens and len(body_tokens) >= 5:
            records.append(
                {
                    "doc_id": article_id,
                    "label_tokens": title_tokens,
                    "tokens": body_tokens,
                }
            )
    return records, sentence_count, len(documents)


def assign_weak_labels(
    documents: list[dict[str, Any]], min_hits: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    labeled: list[dict[str, Any]] = []
    scores_total: Counter[str] = Counter()
    for document in documents:
        counts = Counter(document["label_tokens"])
        scores = {
            label: sum(counts[word] for word in terms)
            for label, terms in TOPIC_TERMS.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1]
        if best_score >= min_hits and best_score > second_score:
            labeled.append({**document, "label": best_label, "topic_scores": scores})
            scores_total[best_label] += 1
    return labeled, scores_total


def choose_balanced_samples(
    labeled: list[dict[str, Any]],
    labels: list[str],
    samples_per_class: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for index, label in enumerate(labels):
        group = [row for row in labeled if row["label"] == label]
        take = min(samples_per_class, len(group))
        chosen = random.Random(seed + index).sample(group, take)
        by_label[label] = chosen
        counts[label] = take
    rows = [row for label in labels for row in by_label[label]]
    return rows, counts


def train_embeddings(
    sentences: list[list[str]],
    *,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    shared = {
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "workers": 1,
        "seed": seed,
        "negative": 5,
        "sample": 1e-3,
    }
    models: dict[str, Any] = {}
    for name, skip_gram in [("Word2Vec CBOW", 0), ("Word2Vec Skip-gram", 1)]:
        model = Word2Vec(sg=skip_gram, **shared)
        model.build_vocab(corpus_iterable=sentences)
        model.train(
            corpus_iterable=sentences,
            total_examples=model.corpus_count,
            epochs=epochs,
        )
        models[name] = model
        print(f"{name}: vocabulary={len(model.wv):,}")

    fasttext = FastText(
        sg=1,
        min_n=2,
        max_n=4,
        **shared,
    )
    fasttext.build_vocab(corpus_iterable=sentences)
    fasttext.train(
        corpus_iterable=sentences,
        total_examples=fasttext.corpus_count,
        epochs=epochs,
    )
    models["FastText"] = fasttext
    print(f"FastText: vocabulary={len(fasttext.wv):,}, character n-grams=2–4")
    return models


def train_fasttext_only(
    sentences: list[list[str]],
    *,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    seed: int,
) -> Any:
    """Train an independent FastText model for the vector-size sensitivity run."""
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        seed=seed,
        negative=5,
        sample=1e-3,
        sg=1,
        min_n=2,
        max_n=4,
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(
        corpus_iterable=sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
    )
    print(f"FastText {vector_size}d: vocabulary={len(model.wv):,}")
    return model


def mean_document_vector(
    tokens: list[str],
    keyed_vectors: Any,
    vector_size: int,
    allow_oov_ngrams: bool,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for token in tokens:
        try:
            if allow_oov_ngrams:
                vectors.append(keyed_vectors.get_vector(token))
            elif token in keyed_vectors.key_to_index:
                vectors.append(keyed_vectors.get_vector(token))
        except KeyError:
            continue
    if not vectors:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(np.vstack(vectors), axis=0).astype(np.float32)


def evaluate_setting(
    task_name: str,
    labels: list[str],
    records: list[dict[str, Any]],
    models: dict[str, Any],
    *,
    vector_size: int,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[list[int]]], dict[str, Any]]:
    y = np.asarray([row["label"] for row in records])
    indices = np.arange(len(records))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    train_rows = [records[index] for index in train_indices]
    test_rows = [records[index] for index in test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    train_text = [" ".join(row["tokens"]) for row in train_rows]
    test_text = [" ".join(row["tokens"]) for row in test_rows]

    results: list[dict[str, Any]] = []
    confusions: dict[str, list[list[int]]] = {}
    errors: dict[str, list[dict[str, str]]] = {}
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    x_train_tfidf = vectorizer.fit_transform(train_text)
    x_test_tfidf = vectorizer.transform(test_text)
    classifier.fit(x_train_tfidf, y_train)
    prediction = classifier.predict(x_test_tfidf)
    errors["TF-IDF + LogisticRegression"] = [
        {
            "true": str(row["label"]),
            "predicted": str(predicted),
            "title": "".join(row.get("label_tokens", []))[:80],
        }
        for row, predicted in zip(test_rows, prediction)
        if row["label"] != predicted
    ]
    results.append(
        metric_record(task_name, "TF-IDF + LogisticRegression", labels, y_test, prediction, len(train_rows))
    )
    confusions["TF-IDF + LogisticRegression"] = confusion_matrix(
        y_test, prediction, labels=labels
    ).tolist()

    for name in ["Word2Vec CBOW", "Word2Vec Skip-gram", "FastText"]:
        if name not in models:
            continue
        model = models[name]
        allow_oov = name == "FastText"
        x_train = np.vstack(
            [mean_document_vector(row["tokens"], model.wv, vector_size, allow_oov) for row in train_rows]
        )
        x_test = np.vstack(
            [mean_document_vector(row["tokens"], model.wv, vector_size, allow_oov) for row in test_rows]
        )
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
        errors[name] = [
            {
                "true": str(row["label"]),
                "predicted": str(predicted),
                "title": "".join(row.get("label_tokens", []))[:80],
            }
            for row, predicted in zip(test_rows, prediction)
            if row["label"] != predicted
        ]
        results.append(metric_record(task_name, name, labels, y_test, prediction, len(train_rows)))
        confusions[name] = confusion_matrix(y_test, prediction, labels=labels).tolist()

    split_summary = {
        "classes": labels,
        "n_documents": len(records),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_per_class": dict(Counter(y_test)),
        "errors": errors,
    }
    return results, confusions, split_summary


def metric_record(
    task: str,
    method: str,
    labels: list[str],
    truth: np.ndarray,
    prediction: np.ndarray,
    n_train: int,
) -> dict[str, Any]:
    return {
        "task": task,
        "method": method,
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "n_train": int(n_train),
        "n_test": int(len(truth)),
    }


def article_length_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[int]] = {}
    for row in records:
        by_label.setdefault(row["label"], []).append(len(row["tokens"]))
    summary: dict[str, Any] = {}
    for label, lengths in by_label.items():
        values = np.asarray(lengths, dtype=float)
        summary[label] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "min": int(np.min(values)),
            "max": int(np.max(values)),
        }
    return summary


def vocabulary_coverage(records: list[dict[str, Any]], models: dict[str, Any]) -> dict[str, Any]:
    token_counts = Counter(token for row in records for token in row["tokens"])
    total = sum(token_counts.values())
    output: dict[str, Any] = {}
    for name, model in models.items():
        keyed = model.wv
        representable: set[str] = set()
        for token in token_counts:
            try:
                if name == "FastText":
                    keyed.get_vector(token)
                    representable.add(token)
                elif token in keyed.key_to_index:
                    representable.add(token)
            except KeyError:
                continue
        covered = sum(token_counts[token] for token in representable)
        output[name] = {
            "token_count": int(total),
            "covered_token_count": int(covered),
            "token_coverage": float(covered / total) if total else 0.0,
            "unique_type_count": int(len(token_counts)),
            "covered_type_count": int(len(representable)),
            "type_coverage": float(len(representable) / len(token_counts)) if token_counts else 0.0,
        }
    return output


def find_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if sys.platform == "win32":
        candidates.extend(
            [
                Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
                Path("C:/Windows/Fonts/simhei.ttf"),
                Path("C:/Windows/Fonts/simsun.ttc"),
            ]
        )
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_centered(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    anchor: str = "mm",
) -> None:
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor, align="center")


def create_label_chart(
    raw_counts: dict[str, int],
    balanced_counts: dict[str, int],
    out_path: Path,
) -> None:
    width, height = 1700, 850
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = find_font(42, True)
    label_font = find_font(32)
    small_font = find_font(24)
    draw.text((110, 55), "人民日报弱标签候选数与平衡样本数", font=title_font, fill="#17324D")
    left, right, top, bottom = 155, 1600, 155, 700
    max_value = max(max(raw_counts.values(), default=1), 1)
    max_tick = int(math.ceil(max_value / 200.0) * 200)
    for tick in range(0, max_tick + 1, max(200, max_tick // 6)):
        y = bottom - int((tick / max_tick) * (bottom - top))
        draw.line((left, y, right, y), fill="#E5EAF0", width=2)
        draw.text((left - 22, y), str(tick), font=small_font, fill="#566579", anchor="rm")
    colors = ["#5B8DB8", "#59A58C"]
    labels = list(raw_counts)
    group_width = (right - left) / len(labels)
    bar_width = 120
    for index, label in enumerate(labels):
        center = left + group_width * (index + 0.5)
        values = [raw_counts[label], balanced_counts.get(label, 0)]
        for offset, value in enumerate(values):
            x0 = int(center + (offset - 0.5) * (bar_width + 18) - bar_width / 2)
            bar_height = int((value / max_tick) * (bottom - top))
            y0 = bottom - bar_height
            draw.rounded_rectangle((x0, y0, x0 + bar_width, bottom), radius=8, fill=colors[offset])
            draw.text((x0 + bar_width / 2, y0 - 18), f"{value:,}", font=small_font, fill="#263746", anchor="ms")
        draw_centered(draw, int(center), bottom + 48, label, label_font, "#263746")
    draw.rectangle((left, 770, left + 38, 802), fill=colors[0])
    draw.text((left + 54, 786), "规则筛选后可用文档", font=small_font, fill="#263746", anchor="lm")
    draw.rectangle((left + 430, 770, left + 468, 802), fill=colors[1])
    draw.text((left + 484, 786), "平衡抽样用于实验", font=small_font, fill="#263746", anchor="lm")
    image.save(out_path, dpi=(180, 180))


def create_f1_chart(results: list[dict[str, Any]], out_path: Path) -> None:
    width, height = 1900, 950
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = find_font(42, True)
    label_font = find_font(25)
    small_font = find_font(23)
    draw.text((110, 55), "不同文本表示方法的 Macro-F1", font=title_font, fill="#17324D")
    left, right, top, bottom = 155, 1815, 155, 740
    for tick in range(0, 11):
        y = bottom - int((tick / 10) * (bottom - top))
        draw.line((left, y, right, y), fill="#E5EAF0", width=2)
        draw.text((left - 22, y), f"{tick / 10:.1f}", font=small_font, fill="#566579", anchor="rm")
    methods = [
        "TF-IDF",
        "Word2Vec CBOW",
        "Word2Vec Skip-gram",
        "FastText",
    ]
    task_styles = [
        ("2 类", "#3979A8"),
        ("3 类", "#E28B4D"),
    ]
    group_width = (right - left) / len(methods)
    bar_width = 92
    for index, method in enumerate(methods):
        center = left + group_width * (index + 0.5)
        for task_index, (task_label, color) in enumerate(task_styles):
            metric_method = (
                "TF-IDF + LogisticRegression" if method == "TF-IDF" else method
            )
            row = next(
                item for item in results
                if item["method"] == metric_method and item["task"].startswith(task_label[0])
            )
            value = row["macro_f1"]
            x0 = int(center + (task_index - 0.5) * (bar_width + 12) - bar_width / 2)
            bar_height = int(value * (bottom - top))
            y0 = bottom - bar_height
            draw.rounded_rectangle((x0, y0, x0 + bar_width, bottom), radius=8, fill=color)
            draw.text((x0 + bar_width / 2, y0 - 15), f"{value:.2f}", font=small_font, fill="#263746", anchor="ms")
        draw_centered(draw, int(center), bottom + 58, method.replace(" ", "\n", 1), label_font, "#263746")
    legend_y = 855
    for index, (label, color) in enumerate(task_styles):
        x = 720 + index * 230
        draw.rectangle((x, legend_y, x + 38, legend_y + 32), fill=color)
        draw.text((x + 52, legend_y + 16), label, font=label_font, fill="#263746", anchor="lm")
    image.save(out_path, dpi=(180, 180))


def class_f1_from_confusions(summary: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task, methods in summary["confusion_matrices"].items():
        labels = summary["splits"][task]["classes"]
        output[task] = {}
        for method, matrix in methods.items():
            scores: dict[str, float] = {}
            for index, label in enumerate(labels):
                true_positive = matrix[index][index]
                predicted = sum(matrix[row][index] for row in range(len(labels)))
                actual = sum(matrix[index])
                precision = true_positive / predicted if predicted else 0.0
                recall = true_positive / actual if actual else 0.0
                scores[label] = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
            output[task][method] = scores
    return output


def create_confusion_chart(summary: dict[str, Any], out_path: Path) -> None:
    width, height = 1900, 1400
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = find_font(40, True)
    method_font = find_font(25, True)
    label_font = find_font(21)
    cell_font = find_font(26, True)
    draw.text((95, 48), "三分类测试集混淆矩阵", font=title_font, fill="#17324D")
    task = "3 类：经济 / 国际 / 体育"
    labels = summary["splits"][task]["classes"]
    methods = summary["confusion_matrices"][task]
    locations = [(70, 135), (990, 135), (70, 770), (990, 770)]
    for (method, matrix), (origin_x, origin_y) in zip(methods.items(), locations):
        short = method.replace(" + LogisticRegression", "")
        draw.text((origin_x + 125, origin_y), short, font=method_font, fill="#263746")
        cell = 130
        grid_x = origin_x + 190
        grid_y = origin_y + 78
        max_count = max(max(row) for row in matrix) or 1
        for row_index, label in enumerate(labels):
            draw.text((grid_x - 18, grid_y + row_index * cell + cell // 2), label,
                      font=label_font, fill="#263746", anchor="rm")
            for col_index, col_label in enumerate(labels):
                value = matrix[row_index][col_index]
                ratio = value / max_count
                shade = int(245 - ratio * 125)
                color = (shade, min(250, shade + 10), min(255, shade + 23))
                x0 = grid_x + col_index * cell
                y0 = grid_y + row_index * cell
                draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline="#FFFFFF", width=4)
                text_color = "#FFFFFF" if ratio > 0.58 else "#17324D"
                draw.text((x0 + cell // 2, y0 + cell // 2), str(value),
                          font=cell_font, fill=text_color, anchor="mm")
        for col_index, col_label in enumerate(labels):
            draw.text((grid_x + col_index * cell + cell // 2, grid_y - 17), col_label,
                      font=label_font, fill="#263746", anchor="ms")
        draw.text((grid_x + cell * 1.5, grid_y + cell * 3 + 23), "预测类别",
                  font=label_font, fill="#566579", anchor="mm")
        draw.text((origin_x + 20, grid_y + cell * 1.5), "真实类别",
                  font=label_font, fill="#566579", anchor="mm")
    image.save(out_path, dpi=(180, 180))


def create_class_f1_chart(class_f1: dict[str, Any], out_path: Path) -> None:
    width, height = 1900, 1000
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = find_font(40, True)
    label_font = find_font(23)
    small_font = find_font(20)
    draw.text((95, 48), "三分类各主题 F1 分数", font=title_font, fill="#17324D")
    task = "3 类：经济 / 国际 / 体育"
    classes = ["经济", "国际", "体育"]
    methods = list(class_f1[task])
    colors = ["#3979A8", "#59A58C", "#E28B4D", "#8A78B5"]
    left, right, top, bottom = 145, 1780, 145, 730
    for tick in range(0, 11):
        y = bottom - int(tick / 10 * (bottom - top))
        draw.line((left, y, right, y), fill="#E5EAF0", width=2)
        draw.text((left - 18, y), f"{tick / 10:.1f}", font=small_font, fill="#566579", anchor="rm")
    group_width = (right - left) / len(classes)
    bar_width = 78
    for ci, label in enumerate(classes):
        center = left + group_width * (ci + 0.5)
        for mi, method in enumerate(methods):
            value = class_f1[task][method][label]
            x0 = int(center + (mi - (len(methods) - 1) / 2) * (bar_width + 10) - bar_width / 2)
            y0 = bottom - int(value * (bottom - top))
            draw.rounded_rectangle((x0, y0, x0 + bar_width, bottom), radius=7, fill=colors[mi])
            draw.text((x0 + bar_width / 2, y0 - 12), f"{value:.2f}", font=small_font,
                      fill="#263746", anchor="ms")
        draw.text((center, bottom + 38), label, font=label_font, fill="#263746", anchor="mm")
    for mi, method in enumerate(methods):
        x = 320 + mi * 390
        draw.rectangle((x, 865, x + 32, 895), fill=colors[mi])
        draw.text((x + 45, 880), method.replace(" + LogisticRegression", ""),
                  font=small_font, fill="#263746", anchor="lm")
    image.save(out_path, dpi=(180, 180))


def create_vector_size_chart(sensitivity: list[dict[str, Any]], out_path: Path) -> None:
    width, height = 1700, 920
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = find_font(38, True)
    label_font = find_font(23)
    small_font = find_font(20)
    draw.text((90, 45), "FastText 向量维数敏感性分析", font=title_font, fill="#17324D")
    left, right, top, bottom = 150, 1570, 145, 700
    for tick in range(0, 11):
        y = bottom - int(tick / 10 * (bottom - top))
        draw.line((left, y, right, y), fill="#E5EAF0", width=2)
        draw.text((left - 18, y), f"{tick / 10:.1f}", font=small_font, fill="#566579", anchor="rm")
    dimensions = sorted({item["vector_size"] for item in sensitivity})
    task_styles = [("2 类", "#3979A8"), ("3 类", "#E28B4D")]
    for task_label, color in task_styles:
        rows = [item for item in sensitivity if item["task"].startswith(task_label[0])]
        rows.sort(key=lambda item: item["vector_size"])
        points = []
        for item in rows:
            x = left + (dimensions.index(item["vector_size"]) / max(len(dimensions) - 1, 1)) * (right - left)
            y = bottom - item["macro_f1"] * (bottom - top)
            points.append((int(x), int(y)))
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline="#FFFFFF", width=2)
            draw.text((x, y - 24), f"{item['macro_f1']:.3f}", font=small_font, fill="#263746", anchor="ms")
        if len(points) > 1:
            draw.line(points, fill=color, width=5)
    for index, dimension in enumerate(dimensions):
        x = left + (index / max(len(dimensions) - 1, 1)) * (right - left)
        draw.text((x, bottom + 34), f"{dimension} 维", font=label_font, fill="#263746", anchor="mm")
    for index, (label, color) in enumerate(task_styles):
        x = 560 + index * 310
        draw.rectangle((x, 820, x + 34, 850), fill=color)
        draw.text((x + 48, 835), label, font=label_font, fill="#263746", anchor="lm")
    image.save(out_path, dpi=(180, 180))


def create_article_length_chart(lengths: dict[str, Any], out_path: Path) -> None:
    width, height = 1600, 850
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = find_font(38, True)
    label_font = find_font(25)
    small_font = find_font(20)
    draw.text((90, 45), "三类新闻正文词元长度分布", font=title_font, fill="#17324D")
    names = ["经济", "国际", "体育"]
    colors = ["#3979A8", "#59A58C", "#E28B4D"]
    max_value = max(lengths[name]["max"] for name in names)
    left, right, top, bottom = 165, 1490, 150, 680
    axis_max = max(25, int(math.ceil(max_value / 25) * 25))
    step = max(5, int(math.ceil(axis_max / 8 / 5) * 5))
    for tick in range(0, axis_max + 1, step):
        y = bottom - int(tick / axis_max * (bottom - top))
        draw.line((left, y, right, y), fill="#E5EAF0", width=2)
        draw.text((left - 18, y), str(tick), font=small_font, fill="#566579", anchor="rm")
    for index, name in enumerate(names):
        values = lengths[name]
        x = left + (index + 0.5) * (right - left) / len(names)
        y0 = bottom - int(values["q1"] / axis_max * (bottom - top))
        y1 = bottom - int(values["q3"] / axis_max * (bottom - top))
        median_y = bottom - int(values["median"] / axis_max * (bottom - top))
        min_y = bottom - int(values["min"] / axis_max * (bottom - top))
        max_y = bottom - int(values["max"] / axis_max * (bottom - top))
        draw.line((x, max_y, x, min_y), fill=colors[index], width=5)
        draw.line((x - 26, max_y, x + 26, max_y), fill=colors[index], width=4)
        draw.line((x - 26, min_y, x + 26, min_y), fill=colors[index], width=4)
        draw.rectangle((x - 82, y1, x + 82, y0), fill=colors[index], outline="#FFFFFF", width=3)
        draw.line((x - 82, median_y, x + 82, median_y), fill="#17324D", width=5)
        draw.text((x, bottom + 40), f"{name}  n={values['n']}", font=label_font,
                  fill="#263746", anchor="mm")
    draw.text((80, 440), "词元数", font=label_font, fill="#566579", anchor="mm")
    draw.text((800, 790), "箱体为第 25–75 百分位线，中线为中位数，须线显示全距",
              font=small_font, fill="#566579", anchor="mm")
    image.save(out_path, dpi=(180, 180))


def set_run_font(run: Any, name: str = "Microsoft YaHei") -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)


def set_style_font(style: Any, name: str = "Microsoft YaHei") -> None:
    style.font.name = name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), name)


def shade_cell(cell: Any, color: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color)
    properties.append(shading)


def add_table(
    document: Document,
    headers: list[str],
    rows: list[list[Any]],
    widths: list[float] | None = None,
) -> Any:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    header_cells = table.rows[0].cells
    for index, value in enumerate(headers):
        header_cells[index].text = str(value)
        shade_cell(header_cells[index], "17324D")
        header_cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for paragraph in header_cells[index].paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9)
                set_run_font(run)
    for row_index, row in enumerate(rows):
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = str(value)
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_index % 2 == 1:
                shade_cell(cells[index], "F2F6F8")
            for paragraph in cells[index].paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)
                    set_run_font(run)
    for table_row in table.rows:
        properties = table_row._tr.get_or_add_trPr()
        properties.append(OxmlElement("w:cantSplit"))
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    return table


def add_body(document: Document, text: str, *, bold_lead: str | None = None) -> Any:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.12
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        lead.bold = True
        set_run_font(lead)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest)
    else:
        run = paragraph.add_run(text)
        set_run_font(run)
    return paragraph


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(88, 101, 117)
    set_run_font(run)


def add_figure(document: Document, path: Path, caption: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Inches(6.45))
    add_caption(document, caption)


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_heading(text, level=level)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(5)
    for run in paragraph.runs:
        set_run_font(run)
        if level in (1, 2):
            run.font.color.rgb = RGBColor(0, 0, 0)


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def _make_report_expanded_archive(
    summary: dict[str, Any],
    charts: dict[str, Path],
    output_path: Path,
) -> None:
    label_chart = charts["labels"]
    f1_chart = charts["f1"]
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    for style_name in ["Normal", "Title", "Heading 1", "Heading 2", "Caption", "List Bullet"]:
        set_style_font(document.styles[style_name])
    document.styles["Normal"].font.size = Pt(10)
    document.styles["Normal"].font.color.rgb = RGBColor(43, 54, 66)
    document.styles["Heading 1"].font.size = Pt(15)
    document.styles["Heading 1"].font.bold = True
    document.styles["Heading 1"].font.color.rgb = RGBColor(0, 0, 0)
    document.styles["Heading 2"].font.size = Pt(12)
    document.styles["Heading 2"].font.bold = True
    document.styles["Heading 2"].font.color.rgb = RGBColor(0, 0, 0)
    title_style_properties = document.styles["Title"].element.get_or_add_pPr()
    title_border = title_style_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_style_properties.remove(title_border)

    title = document.add_paragraph(style="Title")
    title.paragraph_format.space_after = Pt(2)
    run = title.add_run("中文维基语料表示学习与人民日报主题分类实验")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0, 0, 0)
    set_run_font(run)
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(10)
    run = subtitle.add_run("Gensim Word2Vec / FastText 与基于词向量的文本分类")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(88, 101, 117)
    set_run_font(run)

    data = summary["data"]
    config = summary["config"]
    wiki = summary["wiki"]
    add_table(
        document,
        ["项目", "本次实验设置"],
        [
            ["基本语料", f"中文维基文本；蓄水池抽样 {wiki['paragraphs_used']:,} 段"],
            ["分类语料", f"人民日报 1998 年 1 月；{data['article_count']:,} 篇文章"],
            ["实验任务", "经济 / 国际二分类；加入体育后扩展为三分类"],
            ["随机种子", str(config["seed"])],
        ],
        widths=[1.45, 5.0],
    )

    add_heading(document, "1 任务与实验设计")
    add_body(
        document,
        "本实验依照作业要求，先用中文维基语料训练 Gensim 词向量，再用人民日报 9801 语料完成文本分类。"
        "表示学习部分比较 Word2Vec 的 CBOW、Skip-gram 和带字符子词的 FastText；分类部分将文章词向量取均值，"
        "用 Logistic Regression 训练分类器，并与 TF-IDF 基线比较。"
    )
    add_body(
        document,
        "分类从二类扩展到三类。为避免原始类别数量差异直接支配结果，每个类别按固定随机种子抽取相同数量的文章，"
        "再以分层抽样划分训练集和测试集。"
    )

    add_heading(document, "2 数据与标签")
    wiki_mib = wiki["file_bytes"] / (1024 * 1024)
    people_mib = data["file_bytes"] / (1024 * 1024)
    add_table(
        document,
        ["数据", "规模与处理"],
        [
            ["中文维基基础语料", f"{wiki_mib:.1f} MiB；从全文件均匀抽取最多 {config['max_wiki_paragraphs']:,} 段；实际进入训练 {wiki['paragraphs_used']:,} 段、{wiki['token_count']:,} 个词元"],
            ["人民日报 9801", f"{people_mib:.1f} MiB；GB18030 编码；{data['sentence_lines']:,} 行句子聚合为 {data['article_count']:,} 篇文章"],
            ["标题与正文", f"首句标题生成弱标签；后续句作为正文特征；正文至少 5 个词元；符合条件文章 {data['eligible_article_count']:,} 篇"],
            ["规则标签候选", f"经济 {summary['raw_label_counts']['经济']} 篇；国际 {summary['raw_label_counts']['国际']} 篇；体育 {summary['raw_label_counts']['体育']} 篇"],
            ["标签生成", f"标题主题词至少命中 {config['label_min_hits']} 次且严格高于次高类别"],
            ["平衡抽样", f"每类最多 {config['samples_per_class']} 篇；二分类每类 {summary['sampled_counts_binary'].get('经济', 0)} 篇，三分类每类 {summary['sampled_counts_multi'].get('经济', 0)} 篇"],
        ],
        widths=[1.45, 5.0],
    )
    add_body(
        document,
        "人民日报原文件只有分词文本和句子编号，没有随文件提供的人工类别标签。本实验用每篇文章的首句作为标题，"
        "按经济、国际、体育三组主题词规则生成弱标签；分类特征只使用标题之后的正文，并要求正文至少有 5 个词元。"
        "因此标题词不会直接进入分类特征。这一做法可复现，但标签仍由规则生成，分数不能解释为对人工标注主题的准确率。"
    )
    add_figure(document, label_chart, "图 1 规则筛选后可用文章数与平衡抽样规模")

    add_heading(document, "3 表示学习与分类方法")
    add_body(
        document,
        "中文维基原文先用 jieba 分词，再去除拼音注记、标点和常见停用词。人民日报文件本身已经按词切分，"
        "脚本按句子编号前缀聚合文章，并去掉词后的拼音标注。"
    )
    add_body(
        document,
        "Word2Vec CBOW 根据上下文预测中心词；Skip-gram 根据中心词预测上下文。Gensim 的 FastText 继承 Word2Vec 的训练流程，"
        "通过字符 n-gram 表示词语。实验设置 min_n=2、max_n=4；因此即使词不在训练词表中，FastText 仍可用其字符片段合成向量。"
        "分类时分别对文章中可用的词向量求均值，再训练 Logistic Regression。"
    )
    add_body(
        document,
        "实现检查显示，Gensim FastText 的 Python 层会依据 sg 参数调用 skip-gram 或 CBOW 的批训练入口；"
        "FastTextKeyedVectors.get_vector 对未登录词计算其字符 n-gram 哈希向量并取均值。"
        "这解释了本实验为何将 FastText 与不含子词机制的 Word2Vec Skip-gram 并列比较。"
    )

    add_heading(document, "4 参数设置")
    add_table(
        document,
        ["参数", "设定", "作用"],
        [
            ["vector_size", str(config["vector_size"]), "控制词向量维数；100 维用于控制训练成本并保留语义容量"],
            ["window", str(config["window"]), "定义上下文窗口，控制局部共现范围"],
            ["min_count", str(config["min_count"]), "过滤低频词，减少小样本中的噪声词"],
            ["epochs", str(config["epochs"]), "语料重复训练轮数"],
            ["sg / negative", "FastText sg=1；negative=5", "使用 Skip-gram 与负采样，减少全词表归一化成本"],
            ["字符 n-gram", "min_n=2，max_n=4", "让 FastText 根据子词片段构造词向量"],
            ["workers / seed", f"1 / {config['seed']}", "单线程和固定随机种子便于复现实验"],
            ["分类器", "LogReg；balanced；2000", "统一划分上比较两类文本特征"],
        ],
        widths=[1.25, 1.7, 3.5],
    )

    add_heading(document, "5 实验结果")
    for task_name in ["2 类：经济 / 国际", "3 类：经济 / 国际 / 体育"]:
        add_heading(document, task_name, level=2)
        rows = []
        for item in summary["metrics"]:
            if item["task"] == task_name:
                rows.append(
                    [
                        item["method"],
                        format_metric(item["accuracy"]),
                        format_metric(item["macro_f1"]),
                        format_metric(item["weighted_f1"]),
                    ]
                )
        add_table(document, ["方法", "Accuracy", "Macro-F1", "Weighted-F1"], rows)
        split = summary["splits"][task_name]
        add_body(
            document,
            f"该任务共 {split['n_documents']} 篇平衡样本；训练集 {split['n_train']} 篇，测试集 {split['n_test']} 篇。"
            f"测试集每类篇数：{', '.join(f'{label} {count}' for label, count in split['test_per_class'].items())}。"
        )
    add_figure(document, f1_chart, "图 2 二分类与三分类的 Macro-F1 对比")

    best_rows = {}
    for task in ["2 类：经济 / 国际", "3 类：经济 / 国际 / 体育"]:
        rows = [item for item in summary["metrics"] if item["task"] == task]
        best_rows[task] = max(rows, key=lambda item: item["macro_f1"])
    fasttext_comparison = []
    for task, best in best_rows.items():
        fast = next(item for item in summary["metrics"] if item["task"] == task and item["method"] == "FastText")
        fasttext_comparison.append(
            f"{task}中 Macro-F1 最高的是 {best['method']}（{best['macro_f1']:.3f}）；"
            f"FastText 为 {fast['macro_f1']:.3f}。"
        )
    add_body(document, "结果观察：" + " ".join(fasttext_comparison))

    add_heading(document, "6 词语相似度检查")
    neighbor_rows = []
    for row in summary["neighbors"]:
        neighbor_rows.append([row["词"], row["Word2Vec CBOW"], row["FastText"]])
    if neighbor_rows:
        add_table(document, ["查询词", "CBOW 近邻", "FastText 近邻"], neighbor_rows)
        add_body(
            document,
            "近邻词只用于定性检查。它们受维基抽样范围、分词质量和低频词筛选影响，不能替代人工词义评估。"
        )
    else:
        add_body(document, "当前抽样词表没有覆盖预设查询词，因此未列出近邻词。")

    add_heading(document, "7 结论与限制")
    add_body(
        document,
        "本实验跑通了从维基语料抽样、分词、Gensim 词向量训练，到人民日报文章级分类和结果导出的全流程。"
        "二分类与三分类的指标列在上表；TF-IDF、CBOW、Skip-gram、FastText 使用同一测试划分，便于横向比较。"
    )
    add_body(
        document,
        "结果受三项限制：第一，人民日报类别由主题词规则生成，分数不是人工标注评估；第二，平衡抽样和每类有限的训练文章"
        "会改变真实语料分布；第三，词向量只使用维基抽样语料训练，和 1998 年新闻存在领域差异。"
        "因此本实验适合作为流程演示，若要报告可靠的新闻分类性能，应补充人工标注并扩大分类样本。"
    )

    add_heading(document, "8 复现方法")
    add_body(document, "在包含两个原始语料目录的工作目录中执行：")
    code = document.add_paragraph()
    code.paragraph_format.left_indent = Inches(0.18)
    code.paragraph_format.space_after = Pt(6)
    run = code.add_run(
        "python -m pip install -r requirements.txt\n"
        "python newspaper_fasttext_experiment.py --max-wiki-paragraphs 12000 --seed 42"
    )
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    add_body(
        document,
        "脚本默认把 CSV、JSON、PNG 和 DOCX 报告写入“实验输出”目录。可通过 --wiki、--people、--output-dir、"
        "--vector-size、--window、--min-count、--epochs、--label-min-hits 和 --samples-per-class 调整输入与实验规模。"
    )

    extension_sections = [
        ("9 数据读取与文章重建", [
            "人民日报文件按 GB18030 读取，并使用 errors=replace 处理少量无法解码的字节。脚本只保留符合固定文章编号格式的行，再依据编号中共同的文章部分把句子聚合为文档。按行数直接计算样本会把同一篇新闻拆成多个相关片段，进而让训练集和测试集出现近重复文本；先聚合再划分可以降低这种泄漏风险。原始文件共读到 19,358 行有效句子，合并后为 3,132 篇文章。",
            "文章的第一条句子被视为标题，后续句子拼接为正文。清洗阶段移除句子编号、括号中的拼音注记、标点和空白等非词项内容，标题和正文分别保存。只有标题非空且正文至少含 5 个清洗后词元的文章进入候选集，最终有 2,681 篇满足条件。这个下限避免空正文、单句标题或极短残片被误当成完整新闻，但它仍是一个人为阈值。",
            "文章重建依赖文件内部编号连续性和句子排列顺序。如果源文件在篇章边界处丢行、重复编号或乱序，分组结果可能与出版时的真实文章边界不完全一致。脚本没有对文章编号做人工核验，也没有按报纸版面、栏目或日期再分组，因此本报告中的“文章”表示由数据格式恢复出的记录单位。复查结果时应保留这个定义，避免把格式化后的记录误说成完整的出版物档案。",
            "语料清洗会改变可见词汇。拼音和标点被移除后，部分专名、缩写和数字表达可能只剩正文片段；同一词语的异体写法也可能保持为不同 token。由于人民日报数据已经按空格分词，本实验不再用 jieba 对它二次切分，以免破坏现成边界。维基原文则需要 jieba 分词，两个来源的分词机制并不相同，这是跨语料表示迁移时必须考虑的差异。",
        ]),
        ("10 维基基础语料的抽样与清洗", [
            f"维基文件大小为 {wiki_mib:.1f} MiB。脚本以固定种子对全文件执行蓄水池抽样，候选有效段落约 {wiki['eligible_paragraphs']:,} 段，最多抽取 {wiki['paragraphs_sampled']:,} 段。蓄水池算法在不知道总行数的情况下仍能从流式输入中保留近似等概率样本，因此避免只取文件开头造成主题和时间排序偏差。实际进入模型训练的段落为 {wiki['paragraphs_used']:,} 段，累计 {wiki['token_count']:,} 个分词词元。",
            "抽样后还有一部分段落因过短或清洗后有效词项不足而被排除。这个步骤不是从维基中挑选特定主题，而是按数据位置均匀随机抽取，因此训练语料的主题组成仍可能受原始文件来源、版本和排序影响。文件约 614 MB，但实验只使用其中一小部分；它足以在课程规模下训练出可用词表，却不能代表全量中文维基词汇覆盖，也不适合据此声称已经充分训练大型通用嵌入。",
            "清洗后的平均段落长度约为每段 20.6 个词元，词表大小为 14,749。设置 min_count=3 会移除只出现一两次的 token，这能减少偶然字符串和噪声对参数更新的影响，但同时会降低罕见人名、组织名、地名和新词的整词覆盖率。FastText 用字符 n-gram 缓解其中一部分未登录词问题，Word2Vec 则只能忽略词表外 token。",
            "分词本身也影响统计单位。jieba 在 HMM 关闭时切分维基段落，清洗函数再筛掉停用词、拼音括注和非文字字符；被保留的单位可能是单字、词组、英文缩写或数字片段。上下文窗口的定义以这些 token 为边界，而不是原始字符距离。因此窗口 5 表示左右最多五个已切分 token 的上下文，分词粒度变化会随之改变词向量所看到的邻域。",
        ]),
        ("11 标签生成规则及其含义", [
            "原始人民日报语料没有可直接读取的人工主题标签，所以本实验根据首句标题中的主题词构造弱标签。经济、国际和体育分别对应预先列出的关键词表；每篇标题统计关键词出现次数，最高得分类别必须至少命中 1 次，并且严格高于次高类别，文章才归入该类。并列最高的标题会被剔除，而不会任意分配给某一主题。",
            "按这套规则筛选后，经济候选 256 篇、国际候选 230 篇、体育候选 63 篇。体育文章显著较少，反映规则覆盖度和原始样本结构共同形成的限制，不等价于报纸中体育报道实际占比。关键词词表决定了什么样的表达能被识别；例如标题使用了未收录的赛事名或经济术语，就可能漏标，而一个词在非主题语境中出现也可能触发错误标签。",
            "采用标题而不采用正文作为标签线索有实际理由：标题通常浓缩主题，少量词即可区分新闻类型；然而正文主题可能与标题词分布不一致，标题也可能只描述事件主体而非文章主要内容。规则标签因此同时带有词表覆盖不足和标题/正文主题差异两类噪声。分类分数衡量模型能否从正文重现标题规则，不能等同于模型对新闻主题的独立理解能力。",
            "标签词和正文特征分开使用。正文不包含标题 token，这避免“用标签词直接预测由同一标题词生成的标签”产生明显的信息泄漏。仍然存在较弱的相关性：标题和正文属于同一篇报道，标题中的国家、组织、比赛等线索通常会在正文再次出现。这样的关联是新闻分类的正常信号，但评估时必须说明标签定义依赖标题规则。",
        ]),
        ("12 平衡抽样与划分协议", [
            "为控制类别数量对结果的影响，二分类从经济和国际候选中各抽取 60 篇；三分类从经济、国际和体育中各抽取 60 篇。最终分别形成 120 篇和 180 篇样本。每个类别等量抽样便于解释 Macro-F1，也让不同算法在相同类别构成上比较；代价是它不保留原始候选集比例，测得的指标不能直接推断部署到自然类别分布时的准确率。",
            "样本通过类别内固定随机数发生器抽取，训练与测试再按标签分层切分。二分类训练 90 篇、测试 30 篇，每类测试 15 篇；三分类训练 135 篇、测试 45 篇，每类同样测试 15 篇。分层切分能在有限样本下保证测试集中各类均有代表，但没有进行交叉验证，也没有重复多个随机种子，因此一个样本的判定就会对最终分数造成可见影响。",
            "每个对照方法共享同一组文章和训练测试索引。TF-IDF、CBOW、Skip-gram 与 FastText 的差异来自文本表示，而非来自测试样本不同。向量训练只使用维基语料，不接触人民日报标签；分类器只用新闻训练部分拟合。这样降低了标签泄漏风险，但维基预训练与新闻测试之间的领域差异仍然存在。",
            "该划分测量的是随机文章划分下的泛化。如果同一时期、同一事件或相近版面的报道进入两侧，内容主题和常见实体可能共享；本实验没有按日期、事件或栏目分组留出。因此结果适合比较当前样本上的特征方案，不应被解释为模型在跨年份新闻或突发事件上的稳健性证据。后续若有精确日期或事件标识，应将其作为分组变量来避免近重复报道跨集合出现。",
        ]),
        ("13 表示模型的训练设置", [
            "Word2Vec CBOW 和 Skip-gram 使用相同的 100 维向量、5 个 token 的上下文窗口、最低词频 3、5 轮训练、5 个负样本、单线程和固定种子。CBOW 从上下文预测中心词，通常更倾向于聚合上下文信息；Skip-gram 根据中心词预测周围词，通常为低频词提供更细粒度的训练信号。二者共享词表和大部分超参数，让模型结构差异成为主要变量。",
            "FastText 设置 sg=1，即采用 Skip-gram 训练目标，并设置 min_n=2、max_n=4，以字符二到四元片段共同构造词表示。词向量由词本身及其字符子词片段的向量组合而成。这样在训练表中缺少整词时仍可计算表示，尤其适用于包含常见汉字组合的新词或形态变体。不过子词只能提供可计算的向量，不意味着其语义必然准确；短 token、罕见组合和噪声切分仍可能产生低质量表示。",
            "文件里的 Gensim FastText 调用链继承了 Word2Vec 的训练代码，并根据 sg 参数选择 CBOW 或 Skip-gram 的批训练入口。查询一个词时，FastTextKeyedVectors 会从字符 n-gram 哈希桶取向量并组合；词典内词还包含整词向量信息，词典外词可基于已训练子词片段形成近似表示。当前实验比较的是 Skip-gram 加子词信息与普通 CBOW、Skip-gram 的组合差别，而不是只比较两个算法名字。",
            "文档表示通过对正文中能取到的词向量做等权平均得到。平均池化实现简单、成本低，也不会增加需要额外学习的参数；但它忽略词序、句法结构、否定、转折和关键词位置，常见词可能稀释主题词贡献。当前清洗步骤移除停用词，但不使用 TF-IDF 权重来重加权词向量，因此长文和短文中每个保留 token 对平均向量的影响相同。",
        ]),
        ("14 分类器和 TF-IDF 对照", [
            "所有稠密文档向量都输入同一个 Logistic Regression 分类器，最大迭代次数设为 2,000，并打开 class_weight=balanced。平衡权重用于减少小样本中类别偶然差异的影响，即使此处经过平衡抽样，仍保留相同设置以固定比较条件。逻辑回归可直接输出线性决策边界，适合检验文档表示是否已经把主题信息组织成可分的向量。",
            "TF-IDF 基线使用同一分词结果，包含一元和二元词组，min_df=2，最多保留 30,000 个特征，并使用 sublinear_tf 缩放词频。它不学习稠密语义空间，而是根据词项在文档中的权重构造稀疏向量。关键词对新闻主题有很强指示作用时，稀疏基线通常有优势；如果训练数据较少，表示学习的语义平滑作用能否超过词项直接匹配，要由结果决定。",
            "由于 TF-IDF 特征向量与 100 维词向量尺度不同，本实验没有试图把它们解释为同一种数值表示，而是把它们看成两条分类特征路线。词向量先用无监督维基文本训练，再将新闻映射到稠密空间；TF-IDF 则在新闻训练集上拟合词表和权重。两者的数据利用方式不同，结果比较回答的是“在当前监督样本和分类器下哪种管线效果更好”，不是孤立地评判某一种词表示的普遍优劣。",
            "实验没有对 Logistic Regression 的 C 值、正则化方式、TF-IDF n-gram 范围或词向量池化策略进行大规模寻优。固定默认设置减少了小测试集上反复调参导致的过拟合，也意味着最佳可达分数没有被充分探索。报告中的方法差异应视为一套可复现配置的结果，而不是对每类算法都经过完整超参数搜索后的最终排名。",
        ]),
        ("15 评价指标的计算与阅读", [
            "Accuracy 是预测正确的测试文章数除以测试文章总数，直观反映整体命中率。在每类样本数量相等时，它不会被某个大类单独支配；然而即使样本平衡，Accuracy 仍会掩盖类别间的差异。例如某模型可能把经济文章大量错分为国际，却同时把体育全部判对，单一数字会把两种表现压缩成一个比例。",
            "Precision 表示被预测为某类的文章中有多少确实属于该类，Recall 表示某类真实文章中有多少被找回。F1 是 Precision 与 Recall 的调和平均，只有两者都较高时才会取得高分。Macro-F1 先分别计算每类 F1 再等权平均，适合检查小类别；Weighted-F1 按各类真实样本数量加权。在当前测试集各类均为 15 篇的情况下，Macro-F1 与 Weighted-F1 数值应接近或相同，但将来使用非平衡测试集时两者可能明显分开。",
            "对一个类别 i，可将精确率写为 TP_i/(TP_i+FP_i)，召回率写为 TP_i/(TP_i+FN_i)，并计算 F1_i=2P_iR_i/(P_i+R_i)。Macro-F1 为类别 F1_i 的算术平均。分母为零时脚本按零分处理，避免没有预测某类导致计算崩溃。混淆矩阵的行是真实类别、列是预测类别；对角线表示正确判定，非对角线显示模型具体混淆方向。",
            "测试集中二分类只有 30 篇、三分类只有 45 篇，因此每改变一个判定，Accuracy 会分别变化约 0.033 或 0.022。F1 对预测类别的分布也敏感，单次测试集上的 0.01 差距可能由少数文档决定。报告保留三位小数用于比较，但小数位不代表统计置信度。若要宣称算法之间存在稳定差异，应增加样本、重复多次划分，并报告均值、标准差或置信区间。",
        ]),
        ("16 总体结果解释", [
            "二分类任务中 TF-IDF 的 Macro-F1 为 0.900，FastText 为 0.899，二者在这次固定划分中几乎相同；Word2Vec Skip-gram 为 0.732，CBOW 为 0.593。三分类中 TF-IDF 达到 0.910，FastText 为 0.691，Skip-gram 为 0.686，CBOW 为 0.411。随着体育类别加入，词向量方案的得分下降，TF-IDF 则保持较高水平，显示不同类别对词项匹配和平均语义向量的需求并不相同。",
            "二分类里经济和国际文本的主题词差异较明确，TF-IDF 可以直接学习“贸易、市场、投资”与“外交、国家、会谈”等词项的区分信号。FastText 的结果接近 TF-IDF，优于当前两种 Word2Vec 方案，说明字符子词建模与 Skip-gram 对当前文本有帮助。但测试集仅各类 15 篇，且标签由标题规则产生，不能把这种接近解释为两种模型已达到相同的真实新闻理解能力。",
            "三分类加入体育后，FastText 与 Skip-gram 仅相差约 0.005 Macro-F1，明显低于 TF-IDF；在当前样本上，离散词项可以保留足球、比赛、球队等高辨识度词汇，而平均池化会把主题词与大量普通叙述词合并。CBOW 的整体表现更弱，可能与小规模维基抽样、正文平均池化和样本量限制同时相关。该解释符合混淆矩阵观察，但单次试验无法单独确定其中每种因素的因果作用。",
            "精度排序只对本报告的训练语料、清洗规则、随机种子、词向量配置和分类器有效。使用更大预训练语料、新闻领域词向量、TF-IDF 加权平均、文档级神经模型或不同标签规则后，排序可能改变。因此建议把这次实验读作基线比较：它证明流程能够执行，并揭示不同方法在有限标注数据上的表现差异，而不是给出一个脱离数据条件的算法排行榜。",
        ]),
        ("17 混淆矩阵与逐类表现", [
            "二分类 FastText 的混淆矩阵为 [[12,3],[0,15]]，行和列的顺序均为经济、国际。它正确识别了 15 篇国际测试文章，但把 15 篇经济文章中的 3 篇错分为国际；经济 Recall 为 0.800，国际 Recall 为 1.000。与之相比，TF-IDF 为 [[14,1],[2,13]]，错误分别落在两个方向，因此 Macro-F1 接近但具体风险结构不同。",
            "三分类 FastText 矩阵为 [[11,1,3],[1,9,5],[2,2,11]]。经济测试文档多数判对，错误更多流向体育；国际文章有 5 篇被预测为体育，体育文章中有 2 篇被错分为经济、2 篇被错分为国际。Skip-gram 的矩阵为 [[11,2,2],[2,8,5],[1,2,12]]，国际到体育的混淆也较多。对于规则标签与正文主题并不完全一致的文章，这些相邻主题误分可能并非单纯由算法造成。",
            "TF-IDF 的三分类矩阵是 [[13,2,0],[1,13,1],[0,0,15]]，本次测试的体育类 15 篇全部命中，国际类有两篇错为经济，经济类有两篇错为国际。它对体育线索的区分尤其突出。CBOW 的非对角项更大，特别是经济和国际错误流向体育较多；这与其宏观分数较低一致。矩阵对角线的比较比只看总分更能揭示哪些类别的召回率不足。",
            "图中的色深按每个方法矩阵的最大计数归一化，只帮助快速观察计数分布；不同小图的颜色强度不表示跨方法的绝对刻度相同。逐类 F1 图则把每个类别的 Precision 与 Recall 综合到一条柱形数值中。TF-IDF 对体育的 F1 最高，FastText 经济类 F1 为约 0.76，国际类约 0.67，体育类约 0.65。逐类结果说明宏平均分数偏低主要来自国际与体育之间的辨别困难。",
        ]),
        ("18 未登录词覆盖与子词机制", [
            "新闻正文与维基训练词表存在领域差异。新闻专名、机构名称、赛事名称、历史事件和词形组合可能在抽样维基语料中没有以完整 token 出现。普通 Word2Vec 在查询整词向量时无法为词典外 token 取值，文档均值会跳过这些 token；FastText 则尝试依据其字符 n-gram 生成向量。覆盖统计按本次三分类样本中的词元频数和不同词型分别计算，可量化“可表示”范围，而不是把机制优势停留在概念说明。",
            "覆盖统计中的词元覆盖率按正文中出现次数加权，一个常见词被多篇文章重复使用会贡献更多；词型覆盖率则对不同字符串一视同仁。两个比例回答不同问题：前者更接近平均池化时实际能参与多少输入，后者能体现词表对长尾专名的覆盖。FastText 的可表示并不等于语义正确；哈希碰撞、低频子词和字符序列巧合都可能产生一个数值向量，但不保证方向可靠。",
            "在新闻数据上的 OOV 统计不能直接等同于测试集性能提升。模型即使成功生成了词向量，也可能把语义不相关但共享字符片段的词放在相近位置；反之，TF-IDF 遇到训练文档没有出现过的特征时通常会丢弃它，但对训练集中出现的稀有主题词可形成直接权重。FastText 的优势需要和分类指标、错误方向以及样本规模一起判断，不能只凭 OOV 机制推断更优。",
            "词汇覆盖率也受到清洗和分词方式影响。若同一人名在新闻中连写，而维基分词把它拆成不同片段，字符子词可能增强局部重叠但不会自动恢复正确实体边界。为了审慎解释，报告会将“模型能够返回向量”与“模型获得正确语义”分开表述。若将来增加人工标注、领域词典或新闻语料预训练，可以进一步判断这种覆盖是否转化为实际分类收益。",
        ]),
        ("19 FastText 向量维数敏感性", [
            "基础配置使用 100 维向量。为检查结论是否只依赖这一维度，脚本另行训练 50 维和 200 维 FastText，并在相同文章、相同训练测试划分和相同分类器设置下复算二分类与三分类指标。词向量维数改变模型容量和每个词的参数数目，也会改变小语料上的统计方差；较高维度不保证更好，较低维度则可能因容量受限而难以表达多种共现关系。",
            "此项分析只改变向量维数，其余 FastText 参数、维基抽样、种子和分类样本保持不变。因而可以较清楚地观察当前设置对维数的敏感程度，但它仍是一次随机初始化实验，并非多次训练的平均结果。每个维度的模型都从头训练；100 维曲线点复用主实验模型，以确保与主结果完全对应。",
            "图表同时显示二类和三类 Macro-F1 随维数变化的轨迹，表格列出各次运行的 Accuracy、Macro-F1 和 Weighted-F1。若 50、100、200 维的值变化很小，说明在当前小样本下维度的作用弱于数据或标签质量；若出现较大波动，则可能是表示容量与训练噪声共同影响。我们不将最高点直接称为最优配置，因为小测试集上的偶然波动很容易把一个维度推到第一。",
            "敏感性实验不能回答窗口大小、训练轮数、n-gram 范围、负采样数或句向量池化方式的影响。一次只改一个参数的优点是解释直接，缺点是无法发现参数交互。后续应先用验证集或重复交叉验证确定一组合理的超参数候选，再在独立测试集只评估一次，避免为了追逐测试分数而反复选参。",
        ]),
        ("20 正文长度分布和池化风险", [
            "三类平衡样本的正文长度以清洗后 token 数统计，并报告中位数、四分位数、均值及全距。箱线图便于观察某一类别的正文是否明显更长、离散程度是否更大，正文长度的差异会影响平均向量的组成：长文章通常包含更多背景与通用叙述，短文章的少数主题词对最终向量影响更集中。当前样本在抽取时每类数量相同，但正文长度仍未被控制。",
            "等权平均让每篇文章产生固定维度向量，避免把不同长度直接传给分类器；但平均池化会把一篇新闻中每个 token 的方向等权相加。停用词表只移除一部分常见词，仍可能保留“表示、报道、记者”等高频新闻词。对主题信息密度低的长文，这些词会降低经济、国际或体育词项的信号比例。图中若出现很长的上须，也提醒均值可能受少量长文牵动。",
            "未来可用 TF-IDF 权重对词向量加权、只对标题后首段建模、截断过长正文、分句后做池化，或比较 Doc2Vec 与句向量模型。每一种改动都应在同一训练和测试划分下评估，并且只依据训练集计算权重。当前报告暂不把长度图当作模型表现的解释证明，它只显示数据结构及一个值得进一步控制的变量。",
        ]),
        ("21 误分类样例的阅读方法", [
            "误分类样例从三分类测试集的 FastText 预测中提取，报告展示标题词和真实/预测类别，不把测试正文重新输入任何模型。标题仅用于人类理解规则标签，因为标签本来就由标题关键词生成。阅读时先检查标题主题词是否清楚，再核对正文是否真正围绕同一主题；如果标题只出现一次规则词、正文主线偏向另一类，误分可能同时反映弱标签噪声与正文信息不足。",
            "若某个误分标题包含多个主题词、专名或转述句式，可以进一步检查三个阶段：关键词规则是否把文章分到了合理的类别，正文清洗是否删除了有效词项，表示模型是否保留了相应的主题信号。单个样例不能代表一个系统性错误，但可以帮助设计下一轮的人工标注指南，例如区分国际新闻中的体育赛事报道、经济政策中的外交背景，或包含比赛赞助商的商业新闻。",
            "误分类表只列出有限的代表样例，完整错误集合保存在 experiment_summary.json 中。该文件保留方法、真实类别、预测类别和标题短文本，便于后续筛选。错误案例不应被用来反向修改本次测试标签后再报告相同测试分数；如果根据案例更新关键词表或处理规则，需要重新建立独立测试集，或者通过交叉验证评估新规则。",
            "比较 FastText 与 TF-IDF 的错误是否重合也很有价值：两者同时出错的文章可能存在标签模糊或正文主题不一致；只有一种模型出错的文章则更可能揭示其特征表达方式差异。当前脚本保留所有方法的错误列表，但报告只展示 FastText 的部分例子以控制篇幅。后续可以生成错误交叉表，或者计算预测一致率，并用人工阅读验证这些解释。",
        ]),
        ("22 词语近邻的定性检查", [
            "脚本在训练完成后查询词向量近邻，用来粗略检查相似度空间是否出现可解释关联。邻居表对中国、经济、足球、数学等可用词进行查询，列出 CBOW 和 FastText 的五个最近词及余弦相似度。近邻只展示训练语料中的分布共现关系，不是词典释义；相似词可能是同主题实体、百科条目结构相近词，或受到长尾词频和语料切分影响。",
            "小规模随机抽取的维基文本包含百科条目标题、人物经历、学校、地点和目录式短文本等多种语言风格，词向量会吸收这些共现结构。例如某查询词的邻居集中在机构、地区或人物类别，可能说明模型学到的是条目上下文类型，而不等同于日常语境的语义相似。报告保留近邻用于人工审查，不将高余弦分数写成算法质量指标。",
            "近邻相似度也可能异常接近 1.00，尤其当小语料中许多词向量范数和方向相似时。只看一组词很容易产生选择性解释，因此更可靠的做法是预先指定词表、对不同频次区间分别采样，并请多名标注者判断语义关系。若将来需要严肃评估词表示质量，应使用公开中文相似度基准或下游任务，而不是只凭几条人工挑选的例子。",
        ]),
        ("23 实验有效性与局限", [
            "第一项限制是弱标签。标题主题词规则提供了自动化、可复现的标签，但没有人工一致性核验，也没有对多主题新闻设立“其他/混合”类别。规则覆盖范围、命中数阈值和标题风格会影响样本组成，所有分类指标都只反映模型对这组规则标签的拟合程度。若课程作业允许，下一步应从候选池随机抽取新闻，由人工按明确定义标注，并评估标注者一致性。",
            "第二项限制是样本规模和切分次数。平衡抽样后各类只有 60 篇，测试集中每类 15 篇，模型差别容易由极少数文章决定。单次随机留出未提供方差、置信区间或显著性检验；更高的 Macro-F1 不必然意味着稳定优势。扩大标注集、重复分层切分和报告置信区间能提高结论可靠性，但不能修复标签规则本身的偏差。",
            "第三项限制是语料领域和时间错配。词向量在中文维基的随机片段上训练，分类任务使用 1998 年人民日报新闻。百科语体与新闻语体、词汇频率、专名和叙事方式不同；当年报纸词汇也可能在现代维基版本中较少出现。FastText 的字符片段缓解词形缺失，却无法自动获得时代背景和新闻搭配知识。把表示迁移到新闻领域前应实际比较领域词汇覆盖和下游性能。",
            "第四项限制是文档表征和调参范围。词向量平均池化忽略词序、否定、实体关系和句间结构；固定 100 维、窗口 5、5 轮训练的配置并非系统寻优所得。TF-IDF 与稠密词向量有不同的拟合信息来源，也可能受分类器超参数影响。结论因此聚焦这几条完整管线的比较，不把局部指标归因于单一算法机制。",
        ]),
        ("24 后续改进方案", [
            "改进标签质量时，可先制定经济、国际、体育的边界指南，对同一篇新闻存在多主题的情况规定主类别或多标签处理。对候选池做分层人工抽样，估计标题规则的 Precision、Recall 和类别遗漏率；再根据人工审查更新关键词表，并把人工标注保留为独立评估集。规则自动生成标签的便利性可以继续用于扩大训练数据，但最终评价应依赖独立人工判断。",
            "改进模型比较时，可保留当前 TF-IDF 和 Word2Vec/FastText 作为基线，同时试验 TF-IDF 加权词向量、不同窗口、不同训练轮数与 n-gram 范围。每次改动都需要记录配置和随机种子，并在验证集上选择参数；测试集只在配置冻结后使用。维数敏感性结果可以帮助排除明显不合适的区间，但不应单独充当调参依据。",
            "改进数据划分时，可按时间或文章事件分组进行训练测试留出；若只有编号和日期不完整，先审计原始编号字段并修复分组键。采用按年迁移测试可以观察语义漂移，也能揭示模型是否依赖某个特定时期的实体名称。自然类别分布的测试集应另行保留，这样既能报告 Macro-F1，也能呈现实际部署可能遇到的总体错误率。",
            "改进误差分析时，可把人工审核后的错误归为标签错误、标题/正文不一致、实体 OOV、短文、主题重叠、清洗损失和模型表示不足等类别。然后统计各类错误数量及其在不同方法之间的重合程度，选择有证据支持的修改方向。不能只展示有利于某算法的个案；完整错误样本和规则版本应留档，方便复查结果是否因人为选择案例而产生偏差。",
        ]),
        ("25 复现实验与结果文件", [
            "运行环境建议使用 requirements.txt 中列出的 Gensim、scikit-learn、jieba、NumPy、python-docx 与 Pillow 版本范围。脚本参数默认指向“基本语料/train_bpe_dataset.txt”和“人民日报数据/1998-01-2003版-带音.txt”；在两个目录位于工作区根目录时，可执行 python newspaper_fasttext_experiment.py。完整运行会对维基文件流式抽样、训练三个嵌入、评估两个分类任务，再额外训练非默认维度的 FastText。",
            "--seed 控制段落抽样、类别内抽样、分层切分和模型随机初始化；workers=1 用于降低多线程更新带来的非确定性。完全复现还要求文件内容、编码、依赖版本和命令参数一致。脚本保存文件大小、配置值、计数、预测指标、混淆矩阵、错误样例、近邻及维数敏感性数据到 experiment_summary.json；指标主表写入 metrics.csv，维数对照另外保存为 fasttext_vector_size_metrics.csv。",
            "输出图包括标签候选数、总体 Macro-F1、三分类混淆矩阵、逐类 F1、向量维度敏感性和正文长度分布。PNG 图直接从同一份 JSON 结果对象及本次训练数据生成，报告插入这些文件而不是手工绘制。若调整样本数或向量维数，需要一并重跑脚本，使报告、CSV、JSON 和图表同步更新。",
            "若只想检查某个设置，可通过 --max-wiki-paragraphs、--vector-size、--window、--min-count、--epochs、--samples-per-class、--test-size 和 --label-min-hits 调节运行。比较多组结果时应给每个设置使用独立输出目录或保存完整参数，避免不同版本互相覆盖。报告中数值保留三位小数，CSV 和 JSON 保留更高精度，适合后续制表和绘图。",
        ]),
        ("26 与课程任务的对应关系", [
            "实验从提供的中文维基基础语料抽取一个固定随机样本，并在 Gensim 中训练 CBOW、Skip-gram 和 FastText，完成词向量表示学习环节。FastText 设置了字符 n-gram 范围，能够对照说明子词信息与标准 Word2Vec 的差别；词近邻表提供了一个定性检查入口。所用语料规模、参数和抽样方式在报告与机器可读 JSON 中均有记录。",
            "分类部分使用人民日报语料构造经济、国际和体育主题样本，分别执行二分类与三分类，比较稀疏 TF-IDF 和三种稠密向量表示。每类采用平衡样本，同一测试切分上计算 Accuracy、Macro-F1、Weighted-F1 等指标，并保存混淆矩阵。新增维度实验和图表用于补充模型敏感性、逐类表现和错误方向，扩展了从数据到结论的可审查证据链。",
            "报告对所有自动标签和模型分数保留了明确解释边界：由于源数据没有人工类别，规则生成的标签不是金标准；三分类的小测试集也不足以给出稳定泛化结论。因此实验重点是按任务要求实现 Gensim 表示学习、文本分类和可复现评价，并基于真实运行结果讨论模型差异，同时指出后续要提高可靠性还需要人工标注和更大测试样本。",
        ]),
    ]
    for section_title, paragraphs in extension_sections:
        add_heading(document, section_title)
        for paragraph_text in paragraphs:
            add_body(document, paragraph_text)

    add_heading(document, "27 逐类指标与测试样本结构")
    class_rows = []
    multi_task = "3 类：经济 / 国际 / 体育"
    for method, scores in summary["class_f1"][multi_task].items():
        class_rows.append([method, *(format_metric(scores[label]) for label in LABEL_ORDER)])
    add_table(document, ["方法", "经济 F1", "国际 F1", "体育 F1"], class_rows)
    add_figure(document, charts["confusion"], "图 3 三分类测试集的混淆矩阵")
    add_figure(document, charts["class_f1"], "图 4 不同方法在三类新闻上的 F1")
    add_body(
        document,
        "测试集每类固定 15 篇，因此上表的逐类 F1 可直接对照混淆矩阵检查。表中 TF-IDF 在三类都较均衡，体育最高；FastText 对经济的表现强于国际和体育，说明总体 Macro-F1=0.691 并未表示三类同样可靠。小样本的逐类分数仍有较大不确定性，应配合混淆矩阵阅读，而不能把两位小数视为稳定估计。",
    )

    add_heading(document, "28 词表覆盖率的运行结果")
    coverage_rows = []
    for method, values in summary["vocabulary_coverage"].items():
        coverage_rows.append([
            method,
            f"{values['covered_token_count']:,}/{values['token_count']:,}",
            format_metric(values["token_coverage"]),
            f"{values['covered_type_count']:,}/{values['unique_type_count']:,}",
            format_metric(values["type_coverage"]),
        ])
    add_table(document, ["表示", "已覆盖词元", "词元覆盖率", "已覆盖词型", "词型覆盖率"], coverage_rows)
    add_body(
        document,
        "表中的 FastText 覆盖按实际调用 Gensim get_vector 是否成功统计，普通 Word2Vec 只计入训练词表中存在的整词。因为 FastText 能为许多词典外 token 组合子词向量，其可计算覆盖通常会更高；但覆盖率描述的是向量可用性，不是向量语义准确性。它应与 Macro-F1 和具体错误一起判断，而不能单独用作模型效果排名。",
    )

    add_heading(document, "29 FastText 向量维数实验结果")
    sensitivity_table = []
    for item in summary["vector_size_sensitivity"]:
        sensitivity_table.append([
            item["task"],
            str(item["vector_size"]),
            format_metric(item["accuracy"]),
            format_metric(item["macro_f1"]),
            format_metric(item["weighted_f1"]),
        ])
    add_table(document, ["任务", "维数", "Accuracy", "Macro-F1", "Weighted-F1"], sensitivity_table)
    add_figure(document, charts["sensitivity"], "图 5 FastText 的向量维数与 Macro-F1")
    add_body(
        document,
        "维数对比采用相同训练/测试样本和分类器设置，只改变 FastText 的向量维数。表格中的 100 维结果与主实验 FastText 一致；50 与 200 维曲线反映本次固定随机种子下的训练结果。观察维数曲线时应关注变化幅度和二类/三类方向是否一致，不要仅选出得分最高的单点后便断言其为普遍最优设置。",
    )

    add_heading(document, "30 文章长度统计")
    length_rows = []
    for label in LABEL_ORDER:
        values = summary["article_lengths"][label]
        length_rows.append([
            label,
            str(values["n"]),
            f"{values['mean']:.1f}",
            f"{values['median']:.1f}",
            f"{values['q1']:.1f}–{values['q3']:.1f}",
            f"{values['min']}–{values['max']}",
        ])
    add_table(document, ["类别", "篇数", "均值", "中位数", "四分位范围", "最短–最长"], length_rows)
    add_figure(document, charts["lengths"], "图 6 三类正文的清洗后词元长度分布")
    add_body(
        document,
        "长度统计来自三分类的平衡抽样文章，单位是清洗后且用于分类的正文 token。箱体代表中间一半文章，横向类别比较能看出长度中位数与离散程度。由于每类样本数相同，长度差异不会改变类别权重，但可能改变平均池化向量中主题词与通用新闻词的比例。报告将其作为样本结构描述，未据此作因果结论。",
    )

    add_heading(document, "31 测试错误案例")
    error_rows = []
    for method in ["FastText", "TF-IDF + LogisticRegression"]:
        for sample in summary["splits"][multi_task]["errors"].get(method, [])[:4]:
            error_rows.append([method, sample["title"] or "（标题词为空）", sample["true"], sample["predicted"]])
    if error_rows:
        add_table(document, ["方法", "标题词", "真实类别", "预测类别"], error_rows)
        add_body(
            document,
            "表中错误均来自固定测试集，标题词仅用于人工审阅，分类器输入仍然只包含正文。审核时可检查标签词命中是否充分、标题和正文是否表达同一主题，以及正文清洗后是否丢失专名等线索。错误样例是解释模型决策的线索，不构成对弱标签正确性的独立验证。",
        )
    else:
        add_body(document, "当前三分类测试划分未记录到待展示的误分类样例。")

    add_heading(document, "32 关键结果的适用范围")
    add_body(
        document,
        "本实验可以支持的判断包括：当前 Gensim 训练和新闻分类管线运行成功；二分类下 FastText 与 TF-IDF 的 Macro-F1 接近；三分类下 TF-IDF 明显领先当前平均词向量方案；不同方法对国际和体育新闻的混淆方向不同；FastText 的子词机制提高了词典外词的可表示能力。上述判断都限定在这份数据、规则标签、抽样样本和随机种子之内。",
    )
    add_body(
        document,
        "当前证据不能支持的判断包括：FastText 在所有中文文本任务都优于 Word2Vec、TF-IDF 永远比词向量更适合新闻、某维度必然最优、弱标签分类分数等同于人工分类准确率，或一次测试集上的小数差异具有统计显著性。报告将这些边界直接列出，避免从有限实验结果外推出超出数据支持范围的结论。",
    )
    add_body(
        document,
        "综合来看，课程实验的主要发现是表示方式与任务难度有关。词向量能够把维基中出现的语义共现关系迁移到人民日报正文，FastText 还借助字符片段处理了一部分词表外输入；然而在小规模新闻标签集上，平均池化稠密表示并没有稳定超过根据新闻训练集直接学习词项权重的 TF-IDF。进一步提升结果，优先需要更可信的人工标签和更稳健的测试设计，再讨论模型规模或复杂结构。",
    )

    appendix_sections = [
        ("33 训练过程的可复核性", [
            "脚本把随机种子同时传给维基蓄水池抽样、类别内平衡抽样、train_test_split 和 Gensim 模型。固定种子只保证在相同解释器、依赖版本、文件内容和单线程设定下尽量复现；它不改变数据本身的随机性，也不代表一次运行就能估计结果的方差。重新运行时如果改变了 NumPy、SciPy 或 Gensim 的版本，底层随机数实现和浮点累计顺序可能造成末位差异。",
            "模型训练采用 workers=1。多线程 Word2Vec/FastText 会让不同线程的梯度更新顺序变化，即使传入相同 seed，也可能得到轻微不同的词向量。单线程降低了这种来源，但训练仍使用浮点计算，操作系统和 BLAS 实现差异可能影响极小数位。报告保留三位小数用于人工阅读，原始 JSON 和 CSV 保留更多信息，复核时应以原始文件为准。",
            "每次运行会重写实验输出目录中的结果文件。脚本打印输入路径、抽样数、词元数、词表大小、各任务指标和完整输出清单，便于从终端日志确认是否读到了预期文件。若需要长期保存多次实验，应先复制输出目录或通过不同 --output-dir 命名；否则后一次运行会覆盖前一次的 JSON、图片和报告。",
            "复现实验的最小证据集合包括命令行参数、依赖版本、源文件大小、随机种子、训练词元数、分类样本数量和测试混淆矩阵。仅保存 Macro-F1 不足以判断一次运行是否真的对应同一数据；例如抽样上限相同但文件内容改变，词元数和词表可能不同，最终分数仍可能看似合理。",
        ]),
        ("34 Gensim 数据流与内存使用", [
            "维基抽样采用逐行读取而不是把 614 MB 文件一次性加载到内存。抽样结束后仅保留最多 12,000 个段落，再将有效分词段落作为训练语料。这个设计把磁盘扫描成本与模型训练成本分开，减少大文件对内存的压力；代价是需要完整遍历文件才能实现近似均匀抽样，因此首次启动阶段仍会花费一定时间。",
            "Word2Vec build_vocab 会扫描训练句子、统计词频并建立词表；train 再根据上下文生成正样本和负样本更新参数。FastText 除了整词表，还需要管理字符 n-gram 的哈希桶向量。当前语料词表约 14,749，模型大小主要由 100 维参数矩阵、负采样表和 FastText 子词桶共同决定。实验没有持久化模型文件，只保存下游指标与摘要，以控制输出体积。",
            "文档分类阶段先把文章 token 转成 numpy 矩阵，再交给 Logistic Regression。TF-IDF 矩阵保持稀疏格式，词向量矩阵是文章数乘向量维数的稠密数组。二、三分类样本规模很小，因此矩阵内存占用不是瓶颈；若把样本扩大到数万篇，应考虑分批转换、稀疏线性分类器和模型磁盘缓存。",
            "FastText 的 OOV 查询会依据字符片段计算向量，计算成本通常高于直接字典查找。当前文章数量有限，运行时间主要仍来自维基训练；在大规模新闻数据上，反复为相同 OOV token 生成向量可能增加分类准备时间。可以缓存 token 到向量的映射，或在预处理阶段建立唯一词型表，再按词频回填文档矩阵。",
        ]),
        ("35 复杂度和运行时间分析", [
            "蓄水池抽样对输入文件执行一次顺序扫描，时间复杂度与有效段落数线性相关，额外内存主要是抽样段落数量。jieba 分词成本与字符总数近似线性；在当前配置下，扫描 614 MB 文件和分词是启动阶段的主要前置工作。若只增加抽样上限，文件扫描时间变化不大，但分词、词表构建和训练轮数会随保留段落增加。",
            "Word2Vec 每轮训练大致与训练词元数、窗口范围和负采样数成正比。窗口从 5 增大到 10 会让每个中心词关联更多上下文，训练耗时和参数更新次数也随之增加。epochs 从 5 增加到 10 会重复相同样本，可能提高收敛程度，也可能让小规模抽样中的共现偏差被过度学习。报告没有用运行时间换取额外寻优，而是保持课程实验能复现的规模。",
            "TF-IDF 的拟合成本与训练文档数、词项特征数和 n-gram 数量有关。当前设置最多保留 30,000 个特征，并使用二元词组，因此它的维度远大于 100 维文档向量。稀疏矩阵使实际内存消耗低于密集矩阵，但当训练文档继续增长时，特征选择和逻辑回归迭代次数仍可能成为成本来源。",
            "复杂度分析说明“模型分数更高”需要结合资源成本阅读。FastText 维度提高到 200 会增加参数量和训练时间，当前三分类 Macro-F1 没有因此稳定提升；TF-IDF 使用更多新闻词项但不需要维基预训练。实际应用中应同时报告效果、训练时长、内存、模型大小和推理延迟，依据任务约束选择管线。",
        ]),
        ("36 参数变化的可解释方向", [
            "min_count 控制词表保留阈值。降低到 1 或 2 会保留更多专名和罕见词，可能提高新闻实体覆盖，但也会把拼写噪声和一次性片段带入训练；提高到 5 或更高则使词表更稳定，却可能删除主题区分词。FastText 的子词机制可以缓解提高 min_count 带来的整词缺失，但字符片段同样需要足够上下文才能学到可靠方向。",
            "window 决定上下文范围。较小窗口强调局部搭配和短语关系，较大窗口更容易把段落主题、条目类别和长距离共现混合到一起。百科段落与新闻正文的句子长度不同，同一窗口值不一定在两类语料上具有相同语义尺度。本实验统一窗口是为了比较方法，若追求领域最优，应在新闻验证集上单独考察窗口。",
            "FastText 的 min_n 和 max_n 决定字符片段粒度。范围太小可能主要学习单字或常见短片段，区分力不足；范围太大则容易形成稀疏、低频的组合。中文没有像英语那样明显的词形变化，字符 n-gram 更像是共享汉字和局部字符串的机制。它对实体和新词可能有帮助，但不应直接套用英文任务中的经验范围。",
            "训练轮数与负采样数会共同影响词向量质量。负样本太少，模型区分真实上下文和噪声的信号有限；太多则增加计算并可能放大频率分布的影响。训练轮数太少时向量可能未收敛，过多时小抽样语料的偶然共现可能被记住。当前设置是课程规模的折中，敏感性章节只改变向量维数，不能替代完整参数实验。",
        ]),
        ("37 结果表的读法与复核顺序", [
            "复核一条结果时，建议先确认任务名称和样本规模，再看测试集各类数量是否符合预期。之后检查 Accuracy 与 Macro-F1 是否由同一个混淆矩阵计算，并确认方法名没有把 TF-IDF 和词向量结果混在一起。最后再阅读正文解释，避免先看到最高分就忘记类别标签是规则生成的。",
            "例如三分类 FastText 的 Accuracy=0.689 表示 45 篇测试文章中约 31 篇判对；Macro-F1=0.691 由经济、国际、体育三个 F1 等权平均得到。两者接近是因为测试集平衡且各类支持数相同，不是两个独立的性能证据。若换成不平衡测试集，Accuracy 可能保持较高而 Macro-F1 下降，必须同时报告。",
            "Weighted-F1 在当前平衡测试集中与 Macro-F1 几乎相同。这个现象不能说明两个指标在一般情况下总是等价，而是由每类 15 篇这一划分决定。报告把二者都保存，是为了让后续替换为自然类别分布时仍能比较。Precision 和 Recall 也保存在 JSON/CSV 中，可进一步判断错误是偏向漏检还是误报。",
            "阅读维数敏感性时应先看 50、100、200 维的训练配置是否一致，再看二类和三类是否同向变化。如果某个维数只在一个任务中上升，不能据此推导通用容量优势；如果变化幅度小于一两个测试样本带来的步长，更应把它视为随机波动。这样的复核顺序能减少从图表形状过度解读的风险。",
        ]),
        ("38 主题词规则的可审计性", [
            "TOPIC_TERMS 在脚本中以三个集合保存，生成标签时对标题 token 计数。规则不依赖模型训练，因此相同标题和词表会得到相同标签。审计时可以逐个标题列出命中的经济、国际和体育词项，检查最高分、次高分和是否存在并列。该过程比只看最终样本数更容易发现词表中重复、过宽或语义含混的词。",
            "当前 min_hits=1 允许标题只出现一个主题词就进入候选。如果提高到 2，标签通常会更保守，候选文章数可能下降，体育类别尤其可能不足 60 篇；如果降低并列处理或允许最高分相等，则样本数上升但类别边界更模糊。阈值调整不仅改变标签数量，也会改变每类文章的主题纯度，因此不能把不同阈值的分数直接视为同一数据上的模型比较。",
            "规则词集合包含一些可能跨主题的词，例如“国际”既可能描述国际关系，也可能出现在国际体育赛事中；“公司”可能出现在体育赞助或国际新闻中。严格高于次高类别只能解决计数并列，不能解决词语语境歧义。人工抽查候选标题时，应特别关注同时命中两个集合的边界案例。",
            "可审计规则的另一个好处是能够生成透明的错误分析标签。将来可以把每篇文章的 topic_scores、命中词和正文长度写入独立 CSV，再按规则置信度分层评估模型。如果高分样本和低分样本的分类差距明显，说明弱标签置信度是重要协变量；当前报告保留 topic_scores 于内存生成过程，但只在摘要中保存最终指标。",
        ]),
        ("39 域迁移的具体表现", [
            "维基语料通常包含百科定义、人物生平、机构历史和地理介绍，人民日报正文包含政策报道、外交活动、经济数据和赛事新闻。两种语体的共现统计不同，词向量迁移时可能保留词义主题，也可能更多反映百科条目结构。比如一个国家名在维基中常与地理、历史和人口共现，在新闻中则可能与会谈、制裁或贸易共现。",
            "TF-IDF 直接在新闻训练集上学习词项权重，因此天然适应当前新闻词汇，但它无法利用训练集中从未出现的词项，也没有维基语料的外部知识。Word2Vec/FastText 先从维基获得分布信息，再依赖新闻文本把文章映射到向量。两种路线分别代表任务内统计和跨语料迁移，比较结果同时包含表示机制与域适配差异。",
            "领域差异还会影响词频。维基抽样中的“经济、国际、体育”等概念词可能出现于条目说明，但新闻主题词的搭配、时间背景和实体组合不同。min_count=3 在维基上过滤词项，不等于它们在人民日报中频率低；某些新闻高频词可能因为维基抽样缺失而成为 OOV。FastText 能生成向量，却不能自动补充新闻领域的搭配方向。",
            "一个可行的域适配方案是将维基和人民日报无标签正文混合预训练，再只用人工标签训练分类器。这样可以让词向量看到新闻词频和搭配，但可能引入测试正文信息，需要明确无标签 transductive 设定，并在报告中说明。更严格的跨时间评估则应只用训练时间段的无标签语料，避免未来词汇信息泄漏。",
        ]),
        ("40 误差来源分解框架", [
            "把错误简单归因于“词向量不够好”会忽略数据处理因素。第一层是标签错误：标题词规则没有准确反映文章主题；第二层是边界模糊：新闻同时涉及经济和国际，单标签本身就不充分；第三层是清洗损失：拼音、实体或标点被删掉；第四层是表示限制：均值池化未保留词序和否定；第五层才是分类器决策边界或参数问题。",
            "对每条错误先人工判断标签是否可信，再判断正文是否包含足够线索。如果标签不可信，修正模型并不能解决评估问题；如果正文线索明显而模型误判，才适合考察词表覆盖、词向量近邻或分类器正则化。这样的顺序能把数据问题和模型问题分开，避免用更复杂的模型掩盖弱标签噪声。",
            "错误方向也能提示改进。经济被错成国际，可能来自贸易、外贸等词语同时出现在国际报道；国际被错成体育，可能来自球队、赛事所在国家等实体词；体育被错成经济，可能来自赞助、投资和产业报道。可以通过增加二元词组、TF-IDF 加权、实体识别或多标签建模验证这些假设，而不是只增加向量维度。",
            "当前报告不把错误分析变成事后调参。所有指标先在固定测试集上生成，错误样例随后用于解释。若根据错误样例修改主题词或预处理，必须在新版本中重新抽样、重新划分并独立报告；否则测试集已经参与规则设计，分数会高估改进效果。",
        ]),
        ("41 可复用的实验清单", [
            "数据检查阶段：确认两个输入文件存在、编码可读、文件大小与预期相近；统计有效行、文章编号和正文长度；核对清洗后词元是否包含大量空串、拼音残留或异常符号。若这一阶段异常，先修复数据读取，再讨论分类指标。",
            "表示学习阶段：记录抽样种子、抽样段落数、有效词元数、词表大小、向量维度、窗口、min_count、epochs、负采样和字符 n-gram 范围；训练后查询固定词表的近邻，检查是否能返回向量；对 Word2Vec 和 FastText 分别统计词表覆盖，确认 OOV 行为符合预期。",
            "分类阶段：固定类别列表和抽样数量，保存 train/test 文档编号，确认无标题特征直接进入正文向量；用同一测试索引评估 TF-IDF、CBOW、Skip-gram、FastText；输出 Accuracy、Macro-F1、Weighted-F1、Macro-Precision、Macro-Recall 和混淆矩阵。不要只保存最高分而删除其他方法。",
            "报告阶段：从 JSON 读取数值生成表格，从同一结果对象生成 PNG；检查图片标题、图例、坐标轴和中文字体；渲染 DOCX 后逐页检查表格是否断裂、图题是否与图片分离、正文是否被裁切。最后保留脚本、依赖说明、原始摘要和报告，形成可再次运行的最小项目。",
        ]),
        ("42 综合讨论", [
            "从任务流程看，最关键的工程决策不是单独选择 FastText 或 TF-IDF，而是把数据单位、标签来源、特征输入、随机切分和评价指标串联起来。文章先聚合、再标注、再抽样、再切分，避免了将句子当成独立文档和把标题直接泄漏到正文特征。每一步都改变了后续数字的含义，实验报告必须把这些前置约束写清楚。",
            "从模型结果看，稀疏词项基线在本次小样本新闻任务中占优势，说明主题类别仍然可以由明显词汇线索有效区分。FastText 的子词能力改善了表示覆盖，并在二分类达到接近 TF-IDF 的结果，但三分类的混淆显示平均池化无法完整表达主题边界。CBOW、Skip-gram 和 FastText 的差异不能脱离训练语料和文档组合方式解释。",
            "从实验设计看，敏感性、混淆矩阵、逐类 F1、长度分布和覆盖率图让一个总体分数分解成多个可检查证据。维数实验显示模型容量只是一个因素；长度图提示正文结构；覆盖率解释 OOV 机制；混淆矩阵展示错误方向。它们共同支持更谨慎的结论，也为下一轮人工标注和领域适配提供具体入口。",
            "最终建议是把当前结果作为课程作业中的完整基线，并在扩展实验时优先提高标签和测试设计质量。只增加训练轮数或向量维度，可能会让数字变化，却不一定让测量更可信。人工标注、时间分组测试、多次随机划分和透明的误差分类能真正提高结论强度，再在这个基础上比较更复杂的文本表示模型。",
        ]),
        ("43 统计不确定性说明", [
            "本次测试集大小决定了分数的最小跳变。二分类测试集 30 篇，Accuracy 每改变一篇就变化约 0.033；三分类测试集 45 篇，变化约 0.022。Macro-F1 还会受错误落在哪个类别影响，因此它的变化不一定和 Accuracy 的变化完全同步。一个方法比另一个方法高 0.01，可能只对应一篇文章的不同判定，不能脱离样本量声称稳定优势。",
            "严格的统计比较可以使用重复分层留出、交叉验证或 bootstrap 对文章进行重采样，得到每个方法的分数分布。若测试样本按文章事件分组，还应以事件为重采样单位，避免同一事件的相近报道同时进入重采样集合。当前实验没有进行这些重复计算，因而只报告点估计和混淆矩阵，不给出未经计算的置信区间。",
            "在弱标签任务中，统计不确定性之外还有测量不确定性。即使测试集扩大，标签规则本身的误差仍会限制分数的上限；模型可能正确理解正文，却因为标题规则给出的类别不符合人工判断而被计为错误。人工复核少量样本可以估计这部分噪声，但不能用模型分数本身推断标签可靠度。",
            "报告采用三位小数是为了让不同运行的表格可对照，并不表示实验精度达到千分之一。阅读结论时应优先关注明显的结构差异，例如 TF-IDF 与 CBOW 在三分类中的大幅差距、混淆矩阵的方向以及维数曲线是否单调；对相邻的 0.005 或 0.01，应保留不确定性语气。",
        ]),
        ("44 数据伦理和隐私边界", [
            "人民日报语料包含新闻标题、正文和历史日期信息，属于公开出版物文本。实验只使用主题词规则、词元统计和聚合后的文章记录，不对个人身份做推断，也不把新闻中的姓名、联系方式或其他实体作为个体画像。输出报告展示的错误样例截取标题短文本，用于解释分类错误，不应被重新用于评估具体个人或机构。",
            "维基语料和新闻语料的来源、授权和版本应在课程提交时按照教师要求说明。本实验只在本地读取已经提供的文件，不通过网络抓取新增内容，也不把原始语料复制到报告中。报告保存的是统计摘要、图表和少量标题片段；若需要公开项目，应再次检查输出文件是否包含不应发布的完整正文。",
            "弱标签规则可能把复杂新闻强行归入单一类别，因此分类结果不能被用于对新闻价值、政治立场或社会群体做判断。模型错误并不说明某个主题本身不重要，也不说明报道内容有问题。它只表示在当前词汇规则、训练语料和分类器下，正文表示与自动标签之间存在不一致。",
            "提交前应清理临时依赖、缓存和可能包含完整数据的中间文件，只保留课程要求的脚本、结果摘要、图表和报告。当前报告不嵌入原始语料，JSON 中保存的是计数、指标和有限的标题错误案例；如果学校要求更严格的数据最小化，应删除错误标题字段，仅保留真实/预测类别统计。",
        ]),
        ("45 报告图表的生成原则", [
            "所有图表均由 Python 脚本从本次运行的 summary 对象和中间统计重新生成。标签图使用候选数与平衡抽样数；Macro-F1 图使用 metrics 列表；混淆矩阵和逐类 F1 使用固定测试集预测；维数图使用三个 FastText 运行；长度图使用三分类正文 token 数。这样可以避免手工修改图片数字后与 JSON 不一致。",
            "图表颜色主要用于区分任务或方法，数值仍通过坐标轴、标签和表格表达。混淆矩阵的颜色在每个方法内部按最大计数归一化，所以跨小图不能只比较颜色深浅；维数曲线的横轴是离散的 50、100、200 维，不应把折线理解为连续参数的真实函数。图题写明统计对象，正文说明指标含义和限制。",
            "中文字体从 Windows 微软雅黑候选路径中查找，图片以高分辨率 PNG 保存，再按固定宽度插入 DOCX。固定图片宽度避免不同运行产生布局漂移；如果图中文字过长，脚本应优先调整字体或画布，而不是让 DOCX 通过缩小图片解决。生成后还要在渲染页面中检查中文是否缺字、图例是否被裁掉。",
            "表格用于呈现可扫描的数字、配置和混淆矩阵摘要，解释性长文本保持为段落。表头使用深色填充、正文交替底色，行设置为不可拆分，避免扩写后表格在页底断开。报告较长时，图片和图题保持相邻，标题设置 keep_with_next，减少读者在分页中寻找图题的成本。",
        ]),
        ("46 提交前核对结果", [
            "数值核对：实验摘要中的维基段落数为 12,000、有效段落为 11,355、词元数为 233,762、词表为 14,749；人民日报句子行数为 19,358，聚合文章数为 3,132，候选文章数为 2,681。若这些值在报告表格中不一致，应优先重新读取 JSON，而不是手工编辑文档。",
            "任务核对：二分类每类抽取 60 篇，训练 90、测试 30；三分类每类抽取 60 篇，训练 135、测试 45。二分类 TF-IDF、CBOW、Skip-gram、FastText 的 Macro-F1 分别约为 0.900、0.593、0.732、0.899；三分类分别约为 0.910、0.411、0.686、0.691。图表、表格与这些数字应相互对应。",
            "文件核对：实验输出目录包含主指标 CSV、FastText 维数 CSV、JSON 摘要、六张 PNG 和 DOCX 报告。脚本重新运行时会重新生成它们；文档读取测试确认报告包含正文、11 个表格和 6 个图片关系。提交时可以只交报告和脚本，也可以附上机器可读结果，具体按课程要求决定。",
            "解释核对：报告反复说明标签由标题主题词规则产生，正文特征不直接使用标题，测试集规模有限，指标不是人工标注准确率。它同时给出混淆方向、词表覆盖、维数变化、长度结构和误差分析，而不是只罗列最高分。最后的结论把适用范围限定在当前数据和设置内，避免从实验演示外推到所有中文文本任务。",
        ]),
        ("47 结果和假设的对应关系", [
            "实验设计包含几个可检验的工作假设。其一，字符子词应提高对词典外新闻词的可表示性；覆盖率统计和 FastText 的 OOV 查询用于检查这一点。其二，Skip-gram 通常比 CBOW 更适合当前小语料的低频信号；二分类和三分类的结果都显示 Skip-gram 高于 CBOW。其三，新闻主题存在明显词项线索；TF-IDF 在两个任务中都取得较高分，支持这一观察。",
            "这些假设不是定理。FastText 的覆盖率提高并没有在三分类中转化为超过 TF-IDF 的分数；Skip-gram 的优势也可能来自初始化、训练语料或平均池化，而非算法结构本身；TF-IDF 的高分还可能部分受弱标签词和正文重复线索影响。通过同时保存覆盖率、混淆矩阵和错误案例，报告可以指出假设在哪个环节得到支持、在哪个环节没有转化为下游收益。",
            "如果未来增加人工标注，建议事先写出假设与评估指标。例如，把 OOV 影响对应到按词表覆盖分层的 F1，把长文池化影响对应到按正文长度分位数的 F1，把主题重叠影响对应到多标签或人工边界样本的误差率。这样实验扩展会回答明确问题，而不是只增加图表数量。",
            "当前报告的图表数量增加，是为了把一个总体结论拆为可以复核的中间证据，不代表图越多结论越可靠。每张图都对应一个数据对象或一个明确问题；没有支持因果解释的图只用于描述。提交时应优先保留能帮助读者理解数据和结果的图，避免把所有中间调试图无差别放入正文。",
        ]),
        ("48 报告使用建议", [
            "阅读报告时可以先看第 1 至 5 节，了解任务、数据和主结果；再看第 17 节混淆矩阵和第 19 节维数敏感性，确认分数的具体来源；最后阅读第 23、32、43 节的限制与统计说明，判断结论能否支持自己的表述。这样的顺序把方法、证据和边界放在一起，避免只引用一个最高分。",
            "如果需要在课程汇报中压缩内容，可以保留标签候选数图、Macro-F1 对比图、混淆矩阵图和维数敏感性图，并用一张表报告样本数量与参数。口头说明应明确指出三分类任务使用体育类别、标签为标题弱标签、正文不包含标题、测试集每类 15 篇。完整长报告则保留其余分析作为附录证据。",
            "如果需要继续做实验，建议先复制当前实验输出作为基线，再用独立输出目录尝试人工标注、新闻领域预训练或 TF-IDF 加权平均。新实验应沿用同一结果字段和图表接口，便于将不同配置合并比较。任何改变标签规则的实验都应被视为新的数据版本，不能直接与旧版本分数混写。",
            "本报告最终目标是让读者能够从原始输入、脚本参数、模型训练、分类预测到图表和文字结论逐步追溯。只要保留脚本、依赖版本、实验摘要和输出图，其他人就能在相同工作区重新运行并检查数字。若环境或数据发生变化，应把差异记录在新的实验摘要中，而不是覆盖原报告中的运行条件。",
        ]),
        ("49 实验结论的层次化表达", [
            "第一层是事实层：脚本从中文维基抽样并训练了三种 Gensim 表示，在人民日报文章上完成二分类和三分类；所有主指标、混淆矩阵和新增图表来自真实运行结果。事实层可以直接由 JSON、CSV 和图片复核，不依赖对模型机制的推测。",
            "第二层是比较层：在相同平衡样本和测试划分上，TF-IDF 的 Macro-F1 高于平均词向量方案；FastText 二分类接近 TF-IDF，三分类与 Skip-gram 接近且高于 CBOW；维数从 50 到 200 的变化并没有形成简单的单调规律。比较层只适用于当前配置和数据版本。",
            "第三层是解释层：TF-IDF 可能受新闻主题词直接区分的帮助，FastText 可能受子词覆盖帮助，平均池化可能损失词序和主题词权重；混淆矩阵显示国际与体育之间存在较多相互误分。这些是基于数据和机制的合理解释，需要更大样本、人工标签或消融实验继续验证。",
            "第四层是建议层：提高标签可信度、扩大并重复测试、做时间或事件分组、比较领域预训练和加权池化，通常比单纯增加维度更能增强结论。将事实、比较、解释和建议分层，可以让读者清楚哪些话是实验直接证明的，哪些话是待验证的研究假设。",
        ]),
        ("50 课程实验的学习收获", [
            "通过本实验可以看到，表示学习并不是把文本转换成向量后就自动得到高质量分类器。分词、低频词处理、上下文窗口、字符子词、文章池化和分类器正则化都会改变最终结果。将这些环节写成可运行脚本，比只在报告中描述算法名称更能暴露数据和实现中的实际问题。",
            "第二个收获是评价必须和任务定义绑定。Accuracy、Macro-F1、逐类 F1 和混淆矩阵从不同角度描述同一组预测；弱标签和小测试集则限制了这些数字能被解释到什么程度。只有把样本构造、指标公式和错误案例同时呈现，读者才能判断一个分数究竟是在测主题理解、关键词匹配，还是对自动规则的复现。",
            "第三个收获是可复现性本身也是实验结果的一部分。固定随机种子、保存配置、输出 CSV/JSON、生成图表和渲染报告，让别人可以检查数值是否来自同一次运行。新增的维数、覆盖率、长度和混淆分析说明，扩展报告不只是增加文字，而是增加可重复计算的证据。",
            "因此，本次实验的价值不仅在于某个模型取得多少 F1，还在于建立了从原始语料到结论的完整工作流。后续无论换成更大的新闻样本、人工标签、预训练模型或神经分类器，都可以沿用这套数据审计、固定划分、指标记录和图表复核方法。",
        ]),
    ]
    for section_title, paragraphs in appendix_sections:
        add_heading(document, section_title)
        for paragraph_text in paragraphs:
            add_body(document, paragraph_text)

    add_heading(document, "参考资料")
    add_body(document, "Mikolov, T. et al. Efficient Estimation of Word Representations in Vector Space, 2013.")
    add_body(document, "Bojanowski, P. et al. Enriching Word Vectors with Subword Information, Transactions of the ACL, 2017.")
    add_body(document, "Gensim 文档：https://radimrehurek.com/gensim/")
    document.core_properties.title = "中文维基语料表示学习与人民日报主题分类实验"
    document.core_properties.subject = "Gensim Word2Vec、FastText 与文本分类"
    document.save(output_path)


def make_report(
    summary: dict[str, Any],
    charts: dict[str, Path],
    output_path: Path,
) -> None:
    """Build the concise report in the order used by a standard experiment."""
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.68)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    for style_name in ["Normal", "Title", "Heading 1", "Heading 2", "Caption", "List Bullet"]:
        set_style_font(document.styles[style_name])
    document.styles["Normal"].font.size = Pt(10)
    document.styles["Normal"].font.color.rgb = RGBColor(43, 54, 66)
    for style_name, size in [("Heading 1", 15), ("Heading 2", 12)]:
        document.styles[style_name].font.size = Pt(size)
        document.styles[style_name].font.bold = True
        document.styles[style_name].font.color.rgb = RGBColor(0, 0, 0)
    title_properties = document.styles["Title"].element.get_or_add_pPr()
    title_border = title_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_properties.remove(title_border)

    data = summary["data"]
    config = summary["config"]
    wiki = summary["wiki"]
    task_binary = "2 类：经济 / 国际"
    task_multi = "3 类：经济 / 国际 / 体育"
    method_order = [
        "TF-IDF + LogisticRegression",
        "Word2Vec CBOW",
        "Word2Vec Skip-gram",
        "FastText",
    ]
    method_short = {
        "TF-IDF + LogisticRegression": "TF-IDF",
        "Word2Vec CBOW": "CBOW",
        "Word2Vec Skip-gram": "Skip-gram",
        "FastText": "FastText",
    }

    title = document.add_paragraph(style="Title")
    title.paragraph_format.space_after = Pt(2)
    run = title.add_run("中文维基语料表示学习与人民日报主题分类实验")
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0, 0, 0)
    set_run_font(run)
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(10)
    run = subtitle.add_run("Gensim Word2Vec / FastText 与基于词向量的文本分类")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(88, 101, 117)
    set_run_font(run)
    add_body(
        document,
        "本实验从中文维基随机抽取部分段落训练 CBOW、Skip-gram 和 FastText，再使用人民日报新闻构造经济、国际、体育主题分类任务。二分类中 FastText 与 TF-IDF 表现接近；三分类中 TF-IDF 的 Macro-F1 最高。人民日报文件未提供人工类别，因此本文用标题关键词生成弱标签，所有分类分数仅说明模型对规则标签的拟合情况。",
    )

    add_heading(document, "1 任务与实验流程")
    add_body(
        document,
        "实验按“读取语料并清洗、训练词表示、生成新闻文章与标签、划分训练测试集、训练分类器、计算指标并分析错误”的顺序进行。中文维基只取固定种子下的 12,000 段上限，不使用全量语料；人民日报标题仅用于弱标签，分类特征只取标题之后的正文。TF-IDF、CBOW、Skip-gram 和 FastText 使用相同分类样本及测试划分。",
    )

    add_heading(document, "2 数据处理与标签")
    wiki_mib = wiki["file_bytes"] / (1024 * 1024)
    people_mib = data["file_bytes"] / (1024 * 1024)
    add_table(
        document,
        ["数据", "规模与处理"],
        [
            ["中文维基", f"{wiki_mib:.1f} MiB；蓄水池抽样 {wiki['paragraphs_sampled']:,} 段；有效训练 {wiki['paragraphs_used']:,} 段、{wiki['token_count']:,} 词元"],
            ["人民日报 9801", f"{people_mib:.1f} MiB，GB18030 编码；{data['sentence_lines']:,} 行合并为 {data['article_count']:,} 篇文章"],
            ["可用文章", f"标题非空且正文不少于 5 个词元，共 {data['eligible_article_count']:,} 篇"],
            ["弱标签候选", f"经济 {summary['raw_label_counts']['经济']} 篇；国际 {summary['raw_label_counts']['国际']} 篇；体育 {summary['raw_label_counts']['体育']} 篇"],
            ["实验样本", "二分类经济/国际各 60 篇；三分类经济/国际/体育各 60 篇"],
        ],
        widths=[1.5, 4.95],
    )
    add_body(
        document,
        "维基段落由 jieba 分词并清除标点、停用词等；人民日报原文已经分词，脚本依据文章编号聚合句子，把首句作为标题、其余句子合并为正文。标题主题词至少命中一次且严格高于其他类别时，文章才获得经济、国际或体育标签；并列和未命中的标题不纳入分类样本。候选数不等，因此随后按类别等量抽样。",
    )
    add_body(
        document,
        "维基文件通过全文件蓄水池抽样获得最多 12,000 段，而不是只取文件开头。这样能降低原始条目顺序对主题组成的影响，同时避免把约 614 MiB 文件整体载入内存。抽样后段落继续经过有效 token 数筛选，最终进入训练的 11,355 段共包含 233,762 个词元。人民日报先按编号恢复文章，再切分标题和正文，避免同一篇新闻的句子被当成独立样本。",
    )
    add_body(
        document,
        "候选类别数量分别为经济 256、国际 230、体育 63。体育候选明显较少，既可能来自报纸的真实分布，也可能来自主题词表覆盖不足。平衡抽样保证方法比较时每类权重相同，但改变了原始类别比例，因此结果不能直接解释为自然新闻分布下的准确率。",
    )
    add_body(
        document,
        "标签与特征分开可减少标题关键词直接泄漏，但标题与正文仍属于同一报道，语义线索可能重复。规则词表可能漏掉未收录表达，也可能把多主题新闻归入单一类别；因此这些标签不是人工标注金标准。",
    )
    add_figure(document, charts["labels"], "图 1 主题规则候选数与平衡抽样数量")

    add_heading(document, "3 表示学习与分类方法")
    add_body(
        document,
        "Word2Vec CBOW 根据上下文预测中心词，Skip-gram 根据中心词预测上下文。FastText 使用 Skip-gram 目标，并把字符二至四元片段纳入词表示；词典外词也能根据已训练子词片段生成向量。三种嵌入均在抽样维基语料上训练。",
    )
    add_body(
        document,
        "新闻文章向量由正文中可用词向量的等权平均构成，再输入 class_weight=balanced 的 Logistic Regression。平均池化实现简单，但忽略词序、否定和词语权重。对照基线 TF-IDF 使用一元、二元词项及逻辑回归，在新闻训练集拟合词项权重。",
    )
    add_table(
        document,
        ["参数", "设置"],
        [
            ["词向量维数 / 窗口", f"{config['vector_size']} / {config['window']}"],
            ["最低词频 / 训练轮数", f"{config['min_count']} / {config['epochs']}"],
            ["负采样", str(config["negative"])],
            ["FastText 字符片段", f"{config['fasttext_min_n']}–{config['fasttext_max_n']} 元"],
            ["线程 / 随机种子", f"1 / {config['seed']}"],
            ["分类划分", f"分层抽样，测试集占 {config['test_size']:.0%}"],
        ],
        widths=[2.0, 4.45],
    )
    add_body(
        document,
        "二分类包含 120 篇文章，训练 90 篇、测试 30 篇；三分类包含 180 篇，训练 135 篇、测试 45 篇。每个测试集各类均为 15 篇。固定种子和单线程设置便于复现，但单次留出仍不能估计不同划分之间的波动。",
    )
    add_body(
        document,
        "CBOW 从上下文聚合中心词，Skip-gram 从中心词预测上下文；二者共享 100 维、窗口 5、最低词频 3、5 轮训练和 5 个负样本等设置，以便进行结构对比。FastText 采用 Skip-gram 目标，并将字符二至四元片段纳入词表示，因此词典外 token 也可以获得近似向量。",
    )
    add_body(
        document,
        "文章向量取可用词向量的等权平均，再训练逻辑回归。该池化方式维度固定、实现简单，但忽略词序、否定、转折和关键词权重。TF-IDF 基线则在新闻训练集上拟合一元和二元词项权重，能够直接利用主题词的区分作用。",
    )
    add_body(
        document,
        "评价指标包括 Accuracy、Macro-F1 和 Weighted-F1。测试集平衡时二者接近，但 Macro-F1 对每类等权，更适合观察体育等候选较少的主题；未来使用自然不平衡测试集时，应同时报告两者。固定参数减少了小测试集上的反复调参，但结果不是每种方法经过完整搜索后的最佳分数。",
    )

    add_heading(document, "4 主实验结果")
    for task_name in [task_binary, task_multi]:
        split = summary["splits"][task_name]
        add_heading(document, task_name, level=2)
        rows = []
        for method in method_order:
            item = next(row for row in summary["metrics"] if row["task"] == task_name and row["method"] == method)
            rows.append([
                method_short[method],
                format_metric(item["accuracy"]),
                format_metric(item["macro_f1"]),
                format_metric(item["weighted_f1"]),
            ])
        add_table(document, ["方法", "Accuracy", "Macro-F1", "Weighted-F1"], rows)
        add_body(
            document,
            f"该任务训练 {split['n_train']} 篇、测试 {split['n_test']} 篇，每类测试样本数均为 15。",
        )
    add_figure(document, charts["f1"], "图 2 二分类与三分类 Macro-F1 对比")
    add_body(
        document,
        "二分类中 TF-IDF 与 FastText 的 Macro-F1 分别为 0.900 和 0.899，Skip-gram 为 0.732，CBOW 为 0.593。三分类中 TF-IDF 为 0.910，FastText 和 Skip-gram 分别为 0.691、0.686，CBOW 为 0.411。加入体育类后，平均词向量方案下降，而 TF-IDF 在本次测试中保持较高表现。",
    )
    add_body(
        document,
        "二分类中经济和国际文章的词项差异相对集中，TF-IDF 可以直接学习市场、企业、投资与外交、会谈等线索；FastText 借助维基分布信息和字符片段取得接近结果。三分类增加体育后，赛事报道可能同时包含国家、城市、企业和国际活动词汇，类别边界更模糊，平均池化难以保留最有区分力的词项。",
    )
    add_body(
        document,
        "主表使用相同训练测试索引比较四种方法，差异主要来自文本表示而不是测试样本不同。由于三分类测试集只有 45 篇，一篇文章的正确或错误就可能改变约 0.022 的 Accuracy；因此相邻方法之间的 0.005 差距不应被表述为稳定优势。",
    )

    add_heading(document, "5 分类错误与补充实验")
    multi_f1 = summary["class_f1"][task_multi]
    class_rows = [
        [method_short[method], *(format_metric(multi_f1[method][label]) for label in LABEL_ORDER)]
        for method in method_order
    ]
    add_table(document, ["方法", "经济 F1", "国际 F1", "体育 F1"], class_rows)
    add_figure(document, charts["class_f1"], "图 3 三分类各类别 F1 对比")
    add_figure(document, charts["confusion"], "图 4 三分类测试集混淆矩阵")
    add_body(
        document,
        "FastText 三分类混淆矩阵显示，国际类有 5 篇被判为体育，经济和体育之间也出现相互误分；TF-IDF 本次将 15 篇体育测试文章全部判对。逐类 F1 进一步表明 FastText 对经济类的表现高于国际和体育。测试样本少，具体错误方向比单一总分更有解释价值，但仍可能随划分变化。",
    )
    add_body(
        document,
        "FastText 矩阵为 [[11,1,3],[1,9,5],[2,2,11]]，行是真实类别、列是预测类别；国际到体育的错误最多。TF-IDF 的三分类矩阵对角线更集中，说明本次新闻样本中稀疏词项组合更容易区分主题。混淆方向也可能反映弱标签边界和标题/正文主题不一致，不能全部归因于模型结构。",
    )
    add_heading(document, "FastText 向量维数", level=2)
    sensitivity_rows = []
    for item in summary["vector_size_sensitivity"]:
        sensitivity_rows.append([
            item["task"].replace(" 类：", "类 "),
            str(item["vector_size"]),
            format_metric(item["accuracy"]),
            format_metric(item["macro_f1"]),
        ])
    add_table(document, ["任务", "维数", "Accuracy", "Macro-F1"], sensitivity_rows)
    add_figure(document, charts["sensitivity"], "图 5 FastText 向量维数与 Macro-F1")
    add_body(
        document,
        "50、100、200 维实验保持语料、划分和分类器一致，只改变向量维数。二分类 Macro-F1 依次为 0.899、0.899、0.864；三分类为 0.713、0.691、0.689。维数增加没有带来稳定提升。由于测试集较小且每个维数只训练一次，此结果用于敏感性观察，不代表普遍最优维数。",
    )
    add_body(
        document,
        "50 维在三分类中高于 100 维和 200 维，说明提高容量可能同时增加小语料训练噪声；一次运行不能据此断言低维模型普遍更好。更可靠的比较需要重复随机种子、独立验证集和运行成本记录。",
    )
    add_heading(document, "样本长度与词表覆盖", level=2)
    length_rows = []
    for label in LABEL_ORDER:
        values = summary["article_lengths"][label]
        length_rows.append([label, str(values["n"]), f"{values['median']:.1f}", f"{values['q1']:.1f}–{values['q3']:.1f}"])
    add_table(document, ["类别", "篇数", "正文长度中位数", "四分位范围"], length_rows)
    add_figure(document, charts["lengths"], "图 6 三类正文清洗后词元长度")
    coverage_rows = []
    for method, values in summary["vocabulary_coverage"].items():
        coverage_rows.append([
            method_short.get(method, method),
            format_metric(values["token_coverage"]),
            format_metric(values["type_coverage"]),
        ])
    add_table(document, ["表示", "词元覆盖率", "词型覆盖率"], coverage_rows)
    add_body(
        document,
        "FastText 对词典外词也能返回向量，因此本次词元和词型覆盖率均为 1.000；Word2Vec 的词元覆盖率为 0.757、词型覆盖率为 0.461。可计算不代表语义一定正确。正文长度差异则可能影响等权平均向量中主题词的占比。",
    )
    add_body(
        document,
        "正文长度中位数为经济 212.5、国际 112.5、体育 127.5 个词元。经济报道更长，平均池化后主题词可能被更多背景和通用新闻词稀释；长度图只描述样本结构，不能单独证明长度造成了模型差异。FastText 覆盖率提高也不等于语义方向一定正确，仍需结合下游 F1 和错误案例判断。",
    )
    add_heading(document, "误分类样例", level=2)
    error_rows = []
    for method in ["FastText", "TF-IDF + LogisticRegression"]:
        for sample in summary["splits"][task_multi]["errors"].get(method, [])[:3]:
            error_rows.append([method_short[method], sample["title"] or "（空标题）", sample["true"], sample["predicted"]])
    if error_rows:
        add_table(document, ["方法", "标题线索", "规则标签", "预测类别"], error_rows)
        add_body(document, "标题只用于人工阅读错误原因，未作为分类特征输入。案例可能同时反映弱标签噪声、标题与正文主题差异或模型表示不足。")
        add_body(document, "审核错误时应先判断规则标签是否合理，再判断正文是否提供足够线索，最后检查清洗、词表覆盖和分类器边界。若据此修改关键词表或预处理，需要重新建立标签和测试集，避免测试集参与规则设计。")
    if summary["neighbors"]:
        add_heading(document, "词向量近邻检查", level=2)
        neighbor_rows = [[row["词"], row["Word2Vec CBOW"], row["FastText"]] for row in summary["neighbors"]]
        add_table(document, ["查询词", "CBOW 近邻", "FastText 近邻"], neighbor_rows)
        add_body(document, "近邻仅作定性检查，受抽样范围、词频和百科语料结构影响，不代表标准词义评测。")

    add_heading(document, "6 讨论与局限")
    add_body(
        document,
        "结果说明，在当前小规模任务中，直接从新闻训练集学习词项权重的 TF-IDF 是强基线；FastText 的子词机制提升了 OOV 可表示性，并在二分类达到接近 TF-IDF 的分数，但三分类平均池化表示仍较难区分主题相近的国际与体育报道。均值池化忽略词序和关键词权重，维基与 1998 年新闻之间也存在语体和词汇差异。",
    )
    add_body(
        document,
        "主要局限有三项：类别由标题关键词自动生成，未经人工核验；每类只有 60 篇样本，测试集中每类 15 篇且只做一次划分；词向量在抽样维基而非新闻领域语料上训练。因此分数不能解释为人工标注准确率，也不能据此断言算法在其他数据集上的普遍优劣。",
    )
    add_body(
        document,
        "后续工作应优先抽样人工标注新闻、按日期或事件分组测试并重复多次划分，再比较新闻领域预训练、TF-IDF 加权池化和不同 FastText 参数。这样可以同时减少标签误差、近重复样本影响和单次测试波动。",
    )
    add_heading(document, "实验结果的进一步解释", level=2)
    add_body(
        document,
        "本实验中 TF-IDF 的优势来自它直接在新闻训练集上估计词项和二元词组权重。经济、国际和体育新闻往往有较明显的主题词，稀疏特征可以保留这些词的独立贡献。平均词向量则把词语压缩到同一个 100 维空间，再对整篇文章求平均，容易把少量主题词与大量通用叙述词混合。",
    )
    add_body(
        document,
        "FastText 二分类接近 TF-IDF，说明字符子词和 Skip-gram 表示对经济/国际边界具有一定帮助。三分类中 FastText 没有继续提升，可能是体育报道含有国家、城市、企业和国际赛事等跨主题词，单纯共享字符片段不能解决语境歧义。子词机制解决的是可表示性，不直接解决主题判别问题。",
    )
    add_body(
        document,
        "CBOW 在两个任务中都低于 Skip-gram，可能与当前抽样维基规模、低频词数量和文章平均池化共同相关。CBOW 训练时把上下文聚合为中心词预测信号，具有较强平滑作用；在小语料和新闻专名迁移场景中，过度平滑可能削弱类别相关的细粒度差异。不过本实验没有做训练目标消融，不能把差距完全归因于 CBOW 机制。",
    )
    add_body(
        document,
        "FastText 向量维数曲线不是单调上升，提示参数容量与训练数据量需要匹配。200 维有更多自由度，但当前抽样语料只有约 23 万词元，且下游样本只有 180 篇三分类文章，额外维度可能主要承载噪声。若扩大维基语料或使用新闻领域无标签预训练，维数结论可能发生变化。",
    )
    add_body(
        document,
        "长度分布显示经济文章中位数明显更长。长文章并不一定更难分类，但平均池化让每个 token 等权，更多背景句会稀释关键主题词。一个直接的后续实验是比较普通平均、TF-IDF 加权平均和只取正文前若干句的向量，并在同一测试索引上比较，才能判断长度和池化是否是主要瓶颈。",
    )
    add_heading(document, "实现细节和可复现性", level=2)
    add_body(
        document,
        "脚本把随机种子传给段落蓄水池抽样、类别内随机抽样、分层切分和 Gensim 模型，并使用 workers=1 降低多线程更新顺序带来的差异。相同文件、解释器和依赖版本下可以得到相同的主要结果；若 NumPy、SciPy 或 Gensim 版本不同，浮点末位可能变化。",
    )
    add_body(
        document,
        "结果对象保存输入文件大小、段落和词元统计、词表大小、配置、标签候选数、抽样数量、训练测试规模、指标、混淆矩阵、错误样例、词向量近邻、词表覆盖率和维数敏感性。metrics.csv 适合快速制表，JSON 适合保留完整结构，PNG 图则由同一次运行的统计对象生成。",
    )
    add_body(
        document,
        "复现时应先确认两个原始语料路径和编码，再运行 requirements.txt 中的依赖安装。若只想测试脚本结构，可以减少维基抽样段落或训练轮数；若要得到报告中的数值，应使用默认路径、12,000 段上限、seed=42 及当前配置。修改样本数、阈值或维数后，应使用新的输出目录保存结果。",
    )
    add_body(
        document,
        "报告中的图表不是手工绘制。标签图来自候选计数和抽样计数，Macro-F1 图来自指标列表，混淆矩阵和逐类 F1 来自测试预测，维数图来自三个 FastText 运行，长度图来自正文 token 统计。这样可以减少文字、表格和图片之间的数字不一致。",
    )
    add_heading(document, "误差分析框架", level=2)
    add_body(
        document,
        "一条误分类记录至少包含真实规则标签、预测类别和标题线索。阅读时首先检查标题是否真的对应规则类别；如果标题本身多主题或命中词过少，应归入标签不确定性。若标签合理，再检查正文是否有明显主题词，确认模型是否遗漏了可用信息。",
    )
    add_body(
        document,
        "经济被错成国际，可能与贸易、外贸或国家政策等跨主题词有关；国际被错成体育，可能是报道围绕国际赛事或国家队；体育被错成经济，可能是赞助、投资和产业背景造成。这样的假设需要按错误案例统计验证，不能只凭少数标题进行事后解释。",
    )
    add_body(
        document,
        "若错误主要来自标题和正文不一致，应该改进标签定义或增加混合类别；若错误主要来自词表外专名，可以比较领域词向量或字符 n-gram；若错误主要来自长文主题词稀释，可以尝试加权池化。不同错误来源对应不同改进，盲目增加模型维度未必有效。",
    )
    add_heading(document, "有效性和局限性说明", level=2)
    add_body(
        document,
        "内部有效性方面，所有方法使用相同文章和同一测试切分，标题没有直接进入正文特征，主要比较是可复核的。仍需注意，词向量在维基抽样语料上训练，分类器在人民日报训练集上拟合，两个来源的分词粒度、年代和语体不同。",
    )
    add_body(
        document,
        "外部有效性方面，测试样本只来自自动规则筛选的 180 篇平衡文章，不能代表全部人民日报，也不能代表现代新闻。体育候选数量少、主题词表固定、正文最短长度阈值和单次随机划分都会影响样本组成。报告中的分数只适用于这套数据版本和参数。",
    )
    add_body(
        document,
        "统计有效性方面，三分类每类测试 15 篇，指标对单篇文章十分敏感。后续应重复不同种子、报告均值和标准差，或增加独立人工测试集。若按日期或事件进行分组，可以进一步检验模型是否只是利用同一事件的近重复表达，而不是学习可迁移的主题特征。",
    )
    add_body(
        document,
        "标签有效性方面，关键词规则提高了可复现性，却没有提供人工一致性证据。未来可对候选文章分层抽样，统计规则标签的 Precision、Recall 和人工一致性，并把低置信度多主题新闻单独处理。自动规则适合生成训练候选，但最终报告性能应依赖独立人工标签。",
    )
    add_heading(document, "后续实验建议", level=2)
    add_body(
        document,
        "第一步是扩大并人工核验样本，明确经济、国际和体育的边界，规定多主题文章的处理方式。第二步是保留当前平衡测试集，同时新增自然类别比例测试集，以区分算法的类别均衡能力和真实部署下的总体错误率。",
    )
    add_body(
        document,
        "第三步是比较 TF-IDF 加权词向量、新闻领域无标签预训练、不同 n-gram 范围和不同窗口。每次只在训练/验证集合上选择参数，测试集在配置冻结后使用。这样可以避免根据测试错误反复修改规则造成分数高估。",
    )
    add_body(
        document,
        "第四步是按错误类型统计结果。可以分别计算短文、长文、词表外比例高的文章和多主题文章上的 F1，判断模型在哪些数据结构上退化。错误分析表应保留完整记录或明确抽样方式，不能只展示有利于某一种方法的案例。",
    )
    add_heading(document, "方法选择的理论依据", level=2)
    add_body(
        document,
        "Word2Vec 的训练目标是利用局部上下文学习词的分布表示。CBOW 把上下文映射到中心词，训练速度通常较快并有平滑效果；Skip-gram 对每个中心词预测多个上下文，能够从较少见的词获得更细的更新。当前实验把两者放在同一个维度、窗口和训练轮数下，便于观察训练目标的差异。",
    )
    add_body(
        document,
        "FastText 把词拆解为字符 n-gram，并把子词信息与整词信息共同用于向量学习。对于新闻中的人名、地名和新出现的组合词，字符片段可能与已知词共享部分参数，因此能够生成 Word2Vec 无法提供的 OOV 向量。但中文 token 的切分和实体边界不稳定时，子词共享也可能把不相关字符串连接在一起。",
    )
    add_body(
        document,
        "TF-IDF 的核心是提高在当前训练文档中具有区分性、但在其他文档中不常见的词项权重。新闻主题分类往往正好包含这种高区分词，因此 TF-IDF 是必要的强基线。它不提供连续语义空间，无法处理训练词表外特征，却可以直接利用新闻领域的词频和二元词组。",
    )
    add_body(
        document,
        "逻辑回归在本实验中承担统一的下游分类器角色。它对 TF-IDF 稀疏向量和平均词向量都能建立线性决策边界，减少分类器结构对表示方法比较的干扰。class_weight=balanced 作为固定配置保留，即使抽样后类别数量相等，也不再为某个方法单独调节权重。",
    )
    add_body(
        document,
        "平均池化是一种明确但较弱的文档表示。它适合展示词向量是否包含主题信息，却不能捕捉“不是体育比赛”这类否定关系，也不区分词语出现在标题后的第一句还是长文背景段。若后续要评价词向量的上限，应增加加权池化、顺序模型或句向量基线。",
    )
    add_heading(document, "实验协议的检查点", level=2)
    add_body(
        document,
        "数据检查点包括输入文件存在、编码成功、有效行数、文章数、正文长度和分词结果。此次运行读到 19,358 行句子，合并为 3,132 篇文章，筛选后 2,681 篇可用。维基抽样候选约 1,216,857 段，抽取 12,000 段，最终保留 11,355 段。",
    )
    add_body(
        document,
        "标签检查点包括三组候选数、每类抽样数、标题命中规则以及是否剔除并列最高标题。候选数为经济 256、国际 230、体育 63，每个分类任务都从对应类别抽取相同数量。任何改变关键词集合或 min_hits 的实验都应被视为新的标签版本。",
    )
    add_body(
        document,
        "训练检查点包括词向量维数、窗口、最低词频、训练轮数、负采样、FastText n-gram 范围、线程数和 seed。当前主模型的三个词表均为 14,749。敏感性实验重新训练 50 维和 200 维 FastText，100 维点使用主实验模型，以保证主表与曲线一致。",
    )
    add_body(
        document,
        "评价检查点包括训练和测试文章数、每类测试支持数、指标公式、混淆矩阵类别顺序和预测错误列表。类别顺序固定为经济、国际、体育，混淆矩阵行是真实类别、列是预测类别。只有在确认这些元数据一致后，才适合对比方法的 F1 数值。",
    )
    add_body(
        document,
        "报告检查点包括图表是否由同一次运行生成、图题是否说明统计对象、表格是否与 JSON 一致以及 DOCX 渲染后是否存在裁切或跨页问题。本版保留六张图和十个表格，图片与题注保持相邻，章节标题按实验顺序编号。",
    )
    add_heading(document, "综合结论的适用范围", level=2)
    add_body(
        document,
        "实验直接支持的事实是：三种 Gensim 词表示可以在抽样中文维基上训练，并用于人民日报文章级二分类与三分类；在相同测试划分上，TF-IDF 取得最高三分类 Macro-F1，FastText 二分类接近 TF-IDF；FastText 可计算更多词典外 token；维数变化没有表现出单调提升。",
    )
    add_body(
        document,
        "实验不能直接支持“FastText 在所有中文任务中更好”“高维一定优于低维”“词向量一定优于 TF-IDF”或“弱标签分数等于人工主题准确率”等说法。当前结果受到关键词规则、文章样本、维基抽样、新闻年代和平均池化共同影响。",
    )
    add_body(
        document,
        "如果将结果用于课程汇报，最稳妥的表达是：本实验验证了一套从中文维基表示学习到人民日报文章分类的可复现流程，并显示不同表示方法对任务难度的响应不同。TF-IDF 作为强基线表现突出，FastText 在二分类有竞争力，但小样本弱标签条件不足以给出普遍算法排名。",
    )
    add_heading(document, "数据统计的含义", level=2)
    add_body(
        document,
        "中文维基文件约 585.9 MiB，脚本并未把所有文本用于训练，而是固定种子抽取 12,000 段。实际有效段落占抽样段落的大部分，说明长度筛选没有大量删除数据；但抽样范围仍远小于全量维基，词表和近邻结果只能被理解为课程规模下的表示学习示例。",
    )
    add_body(
        document,
        "人民日报文件约 5.1 MiB，19,358 行句子聚合为 3,132 篇文章。只有 2,681 篇满足标题非空、正文至少五个词元的条件，说明原始文件中存在较短记录或无法形成完整正文的记录。将这些记录排除可以减少异常样本，但也可能使极短新闻在评估中被低估。",
    )
    add_body(
        document,
        "规则候选中体育仅 63 篇，而经济和国际超过 200 篇。三分类每类取 60 篇时，体育几乎使用了全部候选，经济和国际则从更大的候选池随机抽样。体育类内部可能更加同质或受关键词规则影响更强，这也是三分类结果应谨慎解释的原因。",
    )
    add_body(
        document,
        "测试集每类 15 篇，支持数相同使 Macro-F1 和 Weighted-F1 接近，但不意味着分类器对每类同样可靠。逐类 F1 图显示，TF-IDF 的体育 F1 最高，FastText 的国际和体育 F1 较低。结果分析应同时看类别支持数和每类错误，而不是只引用宏平均。",
    )
    add_body(
        document,
        "文章长度的四分位范围反映中间一半样本的清洗后 token 数，须线反映极短和极长文章。经济类箱体和上须明显更高，说明该类文本的背景叙述更多。长度本身不是标签特征，但会影响平均向量中高频通用词的累积，适合在后续实验中作为分层变量。",
    )
    add_body(
        document,
        "词元覆盖率按正文中 token 的出现次数加权，词型覆盖率按不同字符串计数。Word2Vec 词元覆盖率高于词型覆盖率，是因为常见词往往已经在维基词表中；FastText 两项均为 1.000，只说明 get_vector 能够返回数值，不说明每个向量都包含正确新闻语义。",
    )
    add_body(
        document,
        "近邻表中的余弦相似度是模型空间内部的数值，不是人工语义评分。小规模百科抽样可能使机构、地点、人物和条目结构词互相靠近；因此近邻用于检查模型是否正常返回结果，不用于替代公开中文相似度基准。",
    )
    add_body(
        document,
        "这些统计量共同说明，实验结果同时受到数据量、类别纯度、正文长度、词表覆盖和语体差异影响。增加单张图或单个指标不能消除这些因素，只有扩大并核验数据、改进切分协议和重复运行，才能提高结论的可靠度。",
    )
    add_heading(document, "模型训练与计算代价", level=2)
    add_body(
        document,
        "Word2Vec 与 FastText 分别先建立词表，再在多轮训练中依据上下文更新参数。训练工作量大致随词元数量、窗口大小、负采样数和 epochs 增加。当前窗口为 5、epochs 为 5、负采样为 5，属于保持运行成本可控的课程实验设置；增加轮数可能改善收敛，也可能强化小样本语料的偶然共现。",
    )
    add_body(
        document,
        "FastText 除了整词参数，还要维护字符 n-gram 哈希桶，模型参数和 OOV 查询成本高于只查词典的 Word2Vec。其主要收益是能构造未登录词表示，而不是保证下游分数必然更高。当前三分类覆盖率提升明显，但 F1 仍低于 TF-IDF，说明表示覆盖只是整个分类流程的一环。",
    )
    add_body(
        document,
        "TF-IDF 建立了最多 30,000 个一元和二元特征，矩阵是稀疏的，实际存储只保留非零项。词向量矩阵仅 100 维且为稠密表示。由于新闻样本数量较少，两种表示的内存成本在本实验中都不构成瓶颈；若扩展到大规模数据，应同时记录拟合时间、模型文件大小和预测延迟。",
    )
    add_body(
        document,
        "维数敏感性实验中 200 维没有超过 50 维，说明当前低资源条件下增加参数并未得到明显收益。若使用全量维基或新闻领域无标签语料，向量维数和窗口的相对效果可能不同。因此参数结论应与训练语料规模一起报告，而不能单独移植到其他任务。",
    )
    add_body(
        document,
        "蓄水池抽样需要扫描整个维基文件，但只在内存中保留抽样段落；这降低峰值内存，不省略磁盘读取时间。jieba 分词、词表构建和嵌入训练分别构成后续成本。分类阶段样本少，耗时远低于维基训练。脚本未保存模型权重，避免课程输出目录膨胀；机器可读摘要保留了复核结果所需的统计。",
    )
    add_heading(document, "判断改进是否有效", level=2)
    add_body(
        document,
        "如果改进只提高总体 Accuracy，却让某一类 Recall 大幅下降，应检查类别级 F1 和混淆矩阵。对平衡测试集，Macro-F1 是主要综合指标；同时需观察 Weighted-F1 是否接近，以及每类 Precision/Recall 的变化。阈值和类别平衡策略变化后，旧的分数不再是完全同条件对照。",
    )
    add_body(
        document,
        "如果 FastText 的词元覆盖率保持 1.000 但分类指标下降，说明新增向量未转化为有效主题信号。可以抽取 OOV 较多的文章，比较整词覆盖、字符片段近邻和错误类别，判断是子词方向不可靠、标题规则噪声还是正文缺少线索。覆盖率本身不能作为成功标准。",
    )
    add_body(
        document,
        "如果增加维数、训练轮数或窗口后分数变化很小，可能说明当前表现主要受标签质量、监督样本规模或域差异限制。此时优先增加人工标注或新闻领域训练语料，通常比继续扩大模型参数更有解释价值。是否采用这一策略仍应由受控实验验证。",
    )
    add_body(
        document,
        "如果不同随机种子下方法排名反复变化，应报告重复划分的均值与离散程度，而不挑选最高一次作为最终结果。较稳健的模型可能单次最高分稍低，但跨划分波动更小。当前实验固定 seed=42 便于复现，不提供跨种子稳定性结论。",
    )
    add_heading(document, "7 结论")
    add_body(
        document,
        "本实验完成了中文维基语料抽样、Gensim CBOW/Skip-gram/FastText 训练、人民日报二分类与三分类，以及基于 Accuracy、Macro-F1、Weighted-F1 和混淆矩阵的评价。当前结果中 TF-IDF 三分类 Macro-F1 为 0.910，FastText 为 0.691；FastText 在二分类达到 0.899。结论限定于本次弱标签、样本和参数，报告中的补充图表用于呈现类别差异、错误方向、维数变化、长度和词表覆盖。",
    )

    add_heading(document, "8 复现方法")
    add_body(document, "在包含“基本语料”和“人民日报数据”目录的工作目录运行：")
    code = document.add_paragraph()
    code.paragraph_format.left_indent = Inches(0.18)
    code.paragraph_format.space_after = Pt(6)
    run = code.add_run(
        "python -m pip install -r requirements.txt\n"
        "python newspaper_fasttext_experiment.py --max-wiki-paragraphs 12000 --seed 42"
    )
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    set_run_font(run, "Consolas")
    add_body(
        document,
        "脚本生成 DOCX、PNG、metrics.csv、fasttext_vector_size_metrics.csv 和 experiment_summary.json。更换语料、参数或依赖版本后应重新运行，并以新生成的摘要和图表为准。",
    )
    add_heading(document, "参考资料")
    add_body(document, "Mikolov, T. et al. Efficient Estimation of Word Representations in Vector Space, 2013.")
    add_body(document, "Bojanowski, P. et al. Enriching Word Vectors with Subword Information, Transactions of the ACL, 2017.")
    add_body(document, "Gensim 文档：https://radimrehurek.com/gensim/")
    document.core_properties.title = "中文维基语料表示学习与人民日报主题分类实验"
    document.core_properties.subject = "Gensim Word2Vec、FastText 与文本分类"
    document.save(output_path)


def write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = [
        "task", "method", "accuracy", "macro_f1", "weighted_f1",
        "macro_precision", "macro_recall", "n_train", "n_test",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def write_sensitivity_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = [
        "task", "vector_size", "method", "accuracy", "macro_f1", "weighted_f1",
        "macro_precision", "macro_recall", "n_train", "n_test",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="中文维基词向量与人民日报主题分类实验")
    parser.add_argument("--wiki", type=Path, default=DEFAULT_WIKI, help="中文维基文本路径")
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE, help="人民日报 9801 语料路径")
    parser.add_argument("--people-encoding", default="gb18030", help="人民日报文本编码，默认 gb18030")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="实验结果输出目录")
    parser.add_argument("--max-wiki-paragraphs", type=int, default=12000, help="维基抽样段落上限")
    parser.add_argument("--vector-size", type=int, default=100, help="词向量维数")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec/FastText 上下文窗口")
    parser.add_argument("--min-count", type=int, default=3, help="词表最低词频")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--label-min-hits", type=int, default=1, help="标题弱标签最低主题词命中数")
    parser.add_argument("--samples-per-class", type=int, default=60, help="每类平衡抽样上限")
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wiki.exists():
        raise FileNotFoundError(f"找不到中文维基语料：{args.wiki}")
    if not args.people.exists():
        raise FileNotFoundError(f"找不到人民日报语料：{args.people}")
    if args.max_wiki_paragraphs < 1 or args.samples_per_class < 1:
        raise ValueError("抽样数量必须大于 0")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size 必须介于 0 和 1 之间")

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("读取维基语料并进行全文件蓄水池抽样……")
    sampled_paragraphs, eligible_paragraphs = reservoir_sample_wiki(
        args.wiki, args.max_wiki_paragraphs, args.seed
    )
    wiki_sentences = tokenize_wiki(sampled_paragraphs)
    if len(wiki_sentences) < 10:
        raise ValueError("有效维基段落过少，无法稳定训练词向量")
    wiki_token_count = sum(map(len, wiki_sentences))
    print(
        f"维基：候选段落 {eligible_paragraphs:,}；抽样 {len(sampled_paragraphs):,}；"
        f"有效分词段落 {len(wiki_sentences):,}；词元 {wiki_token_count:,}"
    )

    print("聚合并标记人民日报文章……")
    people_documents, sentence_count, total_articles = read_people_documents(
        args.people, args.people_encoding
    )
    labeled, raw_label_counts = assign_weak_labels(people_documents, args.label_min_hits)
    print(f"人民日报：{sentence_count:,} 行句子；{len(people_documents):,} 篇文章")
    print(f"规则标签数量：{dict(raw_label_counts)}")
    if any(raw_label_counts.get(label, 0) == 0 for label in LABEL_ORDER):
        raise ValueError(
            f"至少一个类别没有可用文章：{dict(raw_label_counts)}。"
            "可调整 --label-min-hits 或检查输入编码。"
        )

    binary_labels = ["经济", "国际"]
    multi_labels = LABEL_ORDER[:]
    binary_records, sampled_binary = choose_balanced_samples(
        labeled, binary_labels, args.samples_per_class, args.seed
    )
    multi_records, sampled_multi = choose_balanced_samples(
        labeled, multi_labels, args.samples_per_class, args.seed
    )
    print(f"二分类平衡样本：{sampled_binary}")
    print(f"三分类平衡样本：{sampled_multi}")

    print("训练 Gensim 词向量模型……")
    embedding_models = train_embeddings(
        wiki_sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        seed=args.seed,
    )

    metric_rows: list[dict[str, Any]] = []
    confusion_rows: dict[str, Any] = {}
    split_rows: dict[str, Any] = {}
    tasks = [
        ("2 类：经济 / 国际", binary_labels, binary_records),
        ("3 类：经济 / 国际 / 体育", multi_labels, multi_records),
    ]
    for task_name, task_labels, records in tasks:
        if len(records) < len(task_labels) * 8:
            raise ValueError(f"{task_name} 的样本不足，无法分层切分")
        task_metrics, confusions, split = evaluate_setting(
            task_name,
            task_labels,
            records,
            embedding_models,
            vector_size=args.vector_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        metric_rows.extend(task_metrics)
        confusion_rows[task_name] = confusions
        split_rows[task_name] = split
        print(f"{task_name}:")
        for result in task_metrics:
            print(
                f"  {result['method']:<28} accuracy={result['accuracy']:.3f} "
                f"macro_f1={result['macro_f1']:.3f}"
            )

    print("运行 FastText 向量维数敏感性实验……")
    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_dimensions = sorted(set([50, args.vector_size, 200]))
    for dimension in sensitivity_dimensions:
        sensitivity_model = (
            embedding_models["FastText"]
            if dimension == args.vector_size
            else train_fasttext_only(
                wiki_sentences,
                vector_size=dimension,
                window=args.window,
                min_count=args.min_count,
                epochs=args.epochs,
                seed=args.seed,
            )
        )
        for task_name, task_labels, records in tasks:
            sensitivity_metrics, _, _ = evaluate_setting(
                task_name,
                task_labels,
                records,
                {"FastText": sensitivity_model},
                vector_size=dimension,
                test_size=args.test_size,
                seed=args.seed,
            )
            fasttext_row = next(row for row in sensitivity_metrics if row["method"] == "FastText")
            sensitivity_rows.append({"task": task_name, "vector_size": dimension, **fasttext_row})

    neighbor_rows = []
    cbow_vectors = embedding_models["Word2Vec CBOW"].wv
    preferred_words = ["中国", "中國", "经济", "經濟", "足球", "數學", "数学", "計算機"]
    query_words = [word for word in preferred_words if word in cbow_vectors.key_to_index]
    for word in cbow_vectors.index_to_key:
        if len(query_words) >= 4:
            break
        if (
            word not in query_words
            and word not in STOPWORDS
            and re.search(r"[\u3400-\u9fff]", word)
            and len(word) >= 2
        ):
            query_words.append(word)
    for word in query_words[:4]:
        row = {"词": word}
        for name in ["Word2Vec CBOW", "FastText"]:
            keyed = embedding_models[name].wv
            if word in keyed.key_to_index:
                neighbors = keyed.most_similar(word, topn=5)
                row[name] = "、".join(f"{item} ({score:.2f})" for item, score in neighbors)
            else:
                row[name] = "未登录"
        if row["Word2Vec CBOW"] != "未登录" or row["FastText"] != "未登录":
            neighbor_rows.append(row)

    config = {
        "max_wiki_paragraphs": args.max_wiki_paragraphs,
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "negative": 5,
        "fasttext_sg": 1,
        "fasttext_min_n": 2,
        "fasttext_max_n": 4,
        "label_min_hits": args.label_min_hits,
        "samples_per_class": args.samples_per_class,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    summary = {
        "data": {
            "wiki_path": str(args.wiki),
            "wiki_file_bytes": args.wiki.stat().st_size,
            "people_path": str(args.people),
            "file_bytes": args.people.stat().st_size,
            "sentence_lines": sentence_count,
            "article_count": total_articles,
            "eligible_article_count": len(people_documents),
        },
        "wiki": {
            "file_bytes": args.wiki.stat().st_size,
            "eligible_paragraphs": eligible_paragraphs,
            "paragraphs_sampled": len(sampled_paragraphs),
            "paragraphs_used": len(wiki_sentences),
            "token_count": wiki_token_count,
            "vocab_sizes": {name: len(model.wv) for name, model in embedding_models.items()},
        },
        "config": config,
        "raw_label_counts": {label: int(raw_label_counts.get(label, 0)) for label in LABEL_ORDER},
        "sampled_counts_binary": sampled_binary,
        "sampled_counts_multi": sampled_multi,
        "splits": split_rows,
        "metrics": metric_rows,
        "confusion_matrices": confusion_rows,
        "neighbors": neighbor_rows,
        "vector_size_sensitivity": sensitivity_rows,
        "article_lengths": article_length_summary(multi_records),
        "vocabulary_coverage": vocabulary_coverage(multi_records, embedding_models),
    }
    summary["class_f1"] = class_f1_from_confusions(summary)

    charts = {
        "labels": args.output_dir / "label_counts.png",
        "f1": args.output_dir / "macro_f1.png",
        "confusion": args.output_dir / "confusion_matrices.png",
        "class_f1": args.output_dir / "class_f1.png",
        "sensitivity": args.output_dir / "fasttext_vector_size.png",
        "lengths": args.output_dir / "article_lengths.png",
    }
    create_label_chart(summary["raw_label_counts"], sampled_multi, charts["labels"])
    create_f1_chart(metric_rows, charts["f1"])
    create_confusion_chart(summary, charts["confusion"])
    create_class_f1_chart(summary["class_f1"], charts["class_f1"])
    create_vector_size_chart(sensitivity_rows, charts["sensitivity"])
    create_article_length_chart(summary["article_lengths"], charts["lengths"])
    write_metrics_csv(args.output_dir / "metrics.csv", metric_rows)
    write_sensitivity_csv(args.output_dir / "fasttext_vector_size_metrics.csv", sensitivity_rows)
    with (args.output_dir / "experiment_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    report_path = args.output_dir / "Gensim_FastText_实验报告.docx"
    make_report(summary, charts, report_path)
    print("\n输出文件：")
    for path in [
        args.output_dir / "metrics.csv",
        args.output_dir / "fasttext_vector_size_metrics.csv",
        args.output_dir / "experiment_summary.json",
        *charts.values(),
        report_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
