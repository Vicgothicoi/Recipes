"""data_handler 命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_handler.config import DataHandlerConfig
from data_handler.pipeline import DataPipeline
from data_handler.utils.logging_setup import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="菜谱知识图谱数据处理：解析 / 视觉描述 / NER / RE / 导出",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--input",
        "-i",
        type=Path,
        help="输入文件或目录（默认 data/dishes）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", parents=[common], help="仅解析文档")
    p_parse.add_argument("--output", "-o", type=Path, help="中间结果 JSON 目录")

    # caption
    p_cap = sub.add_parser("caption", parents=[common], help="解析并生成图像描述")
    p_cap.add_argument("--no-cache", action="store_true")

    # train-ner
    p_tn = sub.add_parser("train-ner", parents=[common], help="训练 NER (BERT+CRF)")
    p_tn.add_argument("--jsonl", type=Path, help="人工标注 JSONL")
    p_tn.add_argument("--weak", action="store_true", help="从菜谱弱标注训练")
    p_tn.add_argument("--limit", type=int, default=None)
    p_tn.add_argument("--epochs", type=int, default=None)

    # train-re
    p_tr = sub.add_parser("train-re", parents=[common], help="训练关系分类 (BERT)")
    p_tr.add_argument("--jsonl", type=Path, help="人工标注 JSONL")
    p_tr.add_argument("--weak", action="store_true", help="从菜谱弱标注训练")
    p_tr.add_argument("--limit", type=int, default=None)
    p_tr.add_argument("--epochs", type=int, default=None)
    p_tr.add_argument("--negative-ratio", type=float, default=1.0)

    # build-graph
    p_build = sub.add_parser(
        "build-graph", parents=[common], help="全流程：解析→NER→RE→导出 CSV"
    )
    p_build.add_argument("--vision", action="store_true", help="启用视觉描述")
    p_build.add_argument("--no-export", action="store_true")
    p_build.add_argument("--no-merge", action="store_true", help="不合并已有 CSV")
    p_build.add_argument("--limit", type=int, default=None)
    p_build.add_argument("--no-ner", action="store_true")
    p_build.add_argument("--no-re", action="store_true")

    # export
    p_exp = sub.add_parser(
        "export", parents=[common], help="从已有中间 JSON 导出（仅解析结果目录）"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = DataHandlerConfig.from_env()
    input_path = args.input or config.dishes_dir

    if args.command == "parse":
        from data_handler.parsers import parse_directory, parse_file

        if input_path.is_file():
            docs = [parse_file(input_path)]
        else:
            docs = parse_directory(input_path, recursive=True)
        out = args.output or config.intermediate_dir / "parsed"
        out.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            name = Path(doc.source_path).stem + ".json"
            (out / name).write_text(
                __import__("json").dumps(doc.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(f"已解析 {len(docs)} 篇 → {out}")
        return 0

    if args.command == "caption":
        from data_handler.parsers import parse_directory, parse_file
        from data_handler.vision import ImageDescriptor

        if input_path.is_file():
            docs = [parse_file(input_path)]
        else:
            docs = parse_directory(input_path, recursive=True)
        desc = ImageDescriptor(config=config, use_cache=not args.no_cache)
        for doc in docs:
            desc.describe_document(doc)
        print(f"已完成图像描述: {len(docs)} 篇")
        return 0

    if args.command == "train-ner":
        from data_handler.ner import NerTrainer

        trainer = NerTrainer(config=config)
        kw = {}
        if args.epochs:
            kw["epochs"] = args.epochs
        if args.jsonl:
            trainer.train_from_jsonl(args.jsonl, **kw)
        elif args.weak:
            trainer.train_from_weak_labels(
                limit=args.limit,
                output_jsonl=config.ner_annotations_dir / "weak.jsonl",
                **kw,
            )
        else:
            print("请指定 --jsonl 或 --weak", file=sys.stderr)
            return 1
        print(f"NER 训练完成，checkpoint: {config.ner_checkpoint_dir / 'best'}")
        return 0

    if args.command == "train-re":
        from data_handler.re import ReTrainer

        trainer = ReTrainer(config=config)
        kw = {}
        if args.epochs:
            kw["epochs"] = args.epochs
        if args.jsonl:
            trainer.train_from_jsonl(args.jsonl, **kw)
        elif args.weak:
            trainer.train_from_weak_labels(
                limit=args.limit,
                output_jsonl=config.re_annotations_dir / "weak.jsonl",
                negative_ratio=args.negative_ratio,
                **kw,
            )
        else:
            print("请指定 --jsonl 或 --weak", file=sys.stderr)
            return 1
        print(f"RE 训练完成，checkpoint: {config.re_checkpoint_dir / 'best'}")
        return 0

    if args.command == "build-graph":
        pipeline = DataPipeline(config)
        result = pipeline.run(
            input_path,
            run_vision=args.vision,
            run_ner=not args.no_ner,
            run_re=not args.no_re,
            export_csv=not args.no_export,
            merge_existing=not args.no_merge,
            limit=args.limit,
        )
        print(
            f"文档 {len(result.documents)} 篇, "
            f"实体 {result.entities_count}, 关系 {result.relations_count}"
        )
        if result.nodes_csv:
            print(f"nodes: {result.nodes_csv}")
            print(f"relationships: {result.relationships_csv}")
        if result.skipped_paths:
            print(f"跳过已存在: {len(result.skipped_paths)} 篇")
        return 0

    if args.command == "export":
        import json

        from data_handler.schemas import ParsedDocument

        parsed_dir = input_path
        if not parsed_dir.is_dir():
            parsed_dir = config.intermediate_dir / "parsed"
        items = []
        for p in parsed_dir.glob("*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            doc = ParsedDocument.from_dict(data)
            ents = [EntitySpan.from_dict(e) for e in doc.metadata.get("entities", [])]
            rels_data = doc.metadata.get("relations", [])
            from data_handler.schemas import RelationTriple

            rels = [
                RelationTriple(
                    head_text=r["head_text"],
                    head_label=r["head_label"],
                    tail_text=r["tail_text"],
                    tail_label=r["tail_label"],
                    relation_type=r["relation_type"],
                    relation_name=r["relation_name"],
                    score=float(r.get("score", 1.0)),
                    properties=dict(r.get("properties", {})),
                    source_path=r.get("source_path"),
                )
                for r in rels_data
            ]
            items.append((doc, ents, rels))
        from data_handler.export import GraphExporter

        exporter = GraphExporter(config=config)
        export_result = exporter.export_batch(items)
        nodes_p, rels_p = exporter.write_csv(export_result)
        print(f"导出: {nodes_p}, {rels_p}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
