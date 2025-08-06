#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AllenNLP指代消解示例
====================

本示例展示如何使用AllenNLP实现指代消解（Coreference Resolution）。
指代消解是自然语言处理中的一个重要任务，用于识别文本中指向同一实体的不同表达式。

作者: AI Assistant
日期: 2024
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import json
from typing import List, Dict, Any, Tuple
import spacy
from spacy.tokens import Doc, Token

try:
    from allennlp.predictors import Predictor
    from allennlp_models import pretrained
except ImportError:
    print("正在安装必要的依赖包...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "allennlp", "allennlp-models", "spacy"])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "zh_core_web_sm"])
    
    from allennlp.predictors import Predictor
    from allennlp_models import pretrained

class CoreferenceResolver:
    """
    AllenNLP指代消解器
    
    这个类封装了AllenNLP的指代消解功能，提供了易于使用的接口
    """
    
    def __init__(self, model_name: str = "coref-spanbert-large-2021.03.10"):
        """
        初始化指代消解器
        
        Args:
            model_name: 预训练模型名称
        """
        print(f"正在加载模型: {model_name}")
        try:
            # 加载预训练的指代消解模型
            self.predictor = Predictor.from_path(
                f"https://storage.googleapis.com/allennlp-public-models/{model_name}.tar.gz"
            )
        except Exception as e:
            print(f"模型加载失败，使用备用模型: {e}")
            # 备用模型
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
            )
        
        # 加载spaCy模型用于文本处理
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("英文spaCy模型未找到，正在下载...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        print("模型加载完成！")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        对文本进行指代消解预测
        
        Args:
            text: 输入文本
            
        Returns:
            包含聚类信息的字典
        """
        return self.predictor.predict(document=text)
    
    def get_clusters(self, text: str) -> List[List[List[int]]]:
        """
        获取指代聚类
        
        Args:
            text: 输入文本
            
        Returns:
            聚类列表，每个聚类包含多个span的位置信息
        """
        prediction = self.predict(text)
        return prediction.get("clusters", [])
    
    def resolve_coreferences(self, text: str) -> str:
        """
        解析指代关系并返回处理后的文本
        
        Args:
            text: 输入文本
            
        Returns:
            解析后的文本
        """
        prediction = self.predict(text)
        clusters = prediction.get("clusters", [])
        document = prediction["document"]
        
        # 如果没有找到聚类，返回原文
        if not clusters:
            return text
        
        # 构建替换映射
        resolved_text = self._replace_coreferences(document, clusters)
        return resolved_text
    
    def _replace_coreferences(self, document: List[str], clusters: List[List[List[int]]]) -> str:
        """
        替换指代表达式
        
        Args:
            document: 分词后的文档
            clusters: 指代聚类
            
        Returns:
            处理后的文本
        """
        # 创建文档副本
        resolved = document.copy()
        
        # 按照从后往前的顺序处理，避免索引错位
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # 选择最具代表性的mention作为替换词
            representative = self._choose_representative_mention(document, cluster)
            representative_text = " ".join(document[representative[0]:representative[1]+1])
            
            # 替换其他mention
            for mention in sorted(cluster, key=lambda x: x[0], reverse=True):
                if mention != representative:
                    start, end = mention[0], mention[1]
                    # 替换mention
                    for i in range(start, end + 1):
                        if i == start:
                            resolved[i] = representative_text
                        else:
                            resolved[i] = ""
        
        # 重新组合文本
        return " ".join(word for word in resolved if word.strip())
    
    def _choose_representative_mention(self, document: List[str], cluster: List[List[int]]) -> List[int]:
        """
        选择最具代表性的mention
        
        Args:
            document: 分词后的文档
            cluster: 聚类
            
        Returns:
            代表性mention的位置
        """
        # 优先选择最长的mention
        representative = max(cluster, key=lambda x: x[1] - x[0])
        
        # 如果有专有名词，优先选择专有名词
        for mention in cluster:
            mention_text = " ".join(document[mention[0]:mention[1]+1])
            if mention_text[0].isupper() and len(mention_text.split()) > 1:
                representative = mention
                break
        
        return representative
    
    def analyze_coreferences(self, text: str) -> Dict[str, Any]:
        """
        分析指代关系并返回详细信息
        
        Args:
            text: 输入文本
            
        Returns:
            包含详细分析结果的字典
        """
        prediction = self.predict(text)
        clusters = prediction.get("clusters", [])
        document = prediction["document"]
        
        analysis = {
            "original_text": text,
            "tokens": document,
            "num_clusters": len(clusters),
            "clusters": [],
            "resolved_text": self.resolve_coreferences(text)
        }
        
        for i, cluster in enumerate(clusters):
            cluster_info = {
                "cluster_id": i,
                "mentions": [],
                "representative": None
            }
            
            representative = self._choose_representative_mention(document, cluster)
            cluster_info["representative"] = {
                "span": representative,
                "text": " ".join(document[representative[0]:representative[1]+1])
            }
            
            for mention in cluster:
                mention_text = " ".join(document[mention[0]:mention[1]+1])
                cluster_info["mentions"].append({
                    "span": mention,
                    "text": mention_text,
                    "is_representative": mention == representative
                })
            
            analysis["clusters"].append(cluster_info)
        
        return analysis


class ImprovedCoreferenceResolver(CoreferenceResolver):
    """
    改进的指代消解器
    
    基于NeuroSYS的改进方法，解决了传统AllenNLP指代消解的一些问题：
    1. 选择有意义的名词短语作为簇头
    2. 处理回指问题
    3. 处理嵌套mention
    """
    
    def __init__(self, model_name: str = "coref-spanbert-large-2021.03.10"):
        super().__init__(model_name)
    
    def get_span_noun_indices(self, doc: Doc, cluster: List[List[int]]) -> List[int]:
        """
        获取包含名词的span索引
        
        Args:
            doc: spaCy文档对象
            cluster: 聚类
            
        Returns:
            包含名词的span索引列表
        """
        spans = [doc[span[0]:span[1]+1] for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [
            i for i, span_pos in enumerate(spans_pos)
            if any(pos in span_pos for pos in ['NOUN', 'PROPN'])
        ]
        return span_noun_indices
    
    def get_cluster_head(self, doc: Doc, cluster: List[List[int]], noun_indices: List[int]) -> Tuple[Any, List[int]]:
        """
        获取聚类头部（第一个名词短语）
        
        Args:
            doc: spaCy文档对象
            cluster: 聚类
            noun_indices: 名词索引列表
            
        Returns:
            头部span和位置信息
        """
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start:head_end+1]
        return head_span, [head_start, head_end]
    
    def is_containing_other_spans(self, span: List[int], all_spans: List[List[int]]) -> bool:
        """
        检查span是否包含其他span（处理嵌套问题）
        
        Args:
            span: 当前span
            all_spans: 所有span列表
            
        Returns:
            是否包含其他span
        """
        return any([
            s[0] >= span[0] and s[1] <= span[1] and s != span 
            for s in all_spans
        ])
    
    def improved_replace_corefs(self, text: str) -> str:
        """
        改进的指代替换方法
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        # 获取spaCy文档和聚类
        doc = self.nlp(text)
        prediction = self.predict(text)
        clusters = prediction.get("clusters", [])
        
        if not clusters:
            return text
        
        # 将AllenNLP的token索引转换为spaCy的token索引
        spacy_clusters = self._convert_to_spacy_indices(doc, prediction["document"], clusters)
        
        resolved = list(tok.text_with_ws for tok in doc)
        all_spans = [span for cluster in spacy_clusters for span in cluster]
        
        for cluster in spacy_clusters:
            noun_indices = self.get_span_noun_indices(doc, cluster)
            
            if noun_indices:
                mention_span, mention = self.get_cluster_head(doc, cluster, noun_indices)
                
                for coref in cluster:
                    if coref != mention and not self.is_containing_other_spans(coref, all_spans):
                        # 执行替换逻辑
                        coref_start, coref_end = coref[0], coref[1]
                        
                        # 替换文本
                        for i in range(coref_start, coref_end + 1):
                            if i == coref_start:
                                resolved[i] = mention_span.text + resolved[i][len(doc[i].text):]
                            else:
                                resolved[i] = ""
        
        return "".join(resolved)
    
    def _convert_to_spacy_indices(self, doc: Doc, allennlp_tokens: List[str], clusters: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        将AllenNLP的token索引转换为spaCy的token索引
        
        Args:
            doc: spaCy文档
            allennlp_tokens: AllenNLP分词结果
            clusters: AllenNLP聚类
            
        Returns:
            转换后的聚类
        """
        # 简化处理：假设token对应关系是一一对应的
        # 实际应用中可能需要更复杂的对齐算法
        return clusters


def basic_example():
    """基础示例"""
    print("=" * 60)
    print("基础指代消解示例")
    print("=" * 60)
    
    # 创建指代消解器
    resolver = CoreferenceResolver()
    
    # 测试文本
    texts = [
        "John is a software engineer. He works at Google. John loves his job because it challenges him.",
        "The cat sat on the mat. It was very comfortable. The cat purred loudly.",
        "Mary went to the store. She bought some apples. Mary gave them to her children.",
        "The company announced its new product. It will be launched next month. The product promises to revolutionize the industry."
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n示例 {i}:")
        print(f"原文: {text}")
        
        # 分析指代关系
        analysis = resolver.analyze_coreferences(text)
        
        print(f"聚类数量: {analysis['num_clusters']}")
        
        for cluster in analysis['clusters']:
            print(f"  聚类 {cluster['cluster_id']}:")
            print(f"    代表性mention: {cluster['representative']['text']}")
            for mention in cluster['mentions']:
                status = " (代表)" if mention['is_representative'] else ""
                print(f"    - {mention['text']}{status}")
        
        print(f"解析后: {analysis['resolved_text']}")


def advanced_example():
    """高级示例"""
    print("\n" + "=" * 60)
    print("高级指代消解示例")
    print("=" * 60)
    
    # 创建改进的指代消解器
    resolver = ImprovedCoreferenceResolver()
    
    # 复杂测试文本
    complex_texts = [
        """
        Barack Obama was the 44th President of the United States. He served two terms from 2009 to 2017. 
        Obama was born in Hawaii and spent his early years there. Before becoming president, he worked as 
        a community organizer and later as a senator. His wife, Michelle Obama, is also a well-known figure. 
        She served as First Lady during his presidency. The Obama family has two daughters.
        """,
        """
        Apple Inc. is one of the world's largest technology companies. The company was founded by Steve Jobs, 
        Steve Wozniak, and Ronald Wayne in 1976. It is headquartered in Cupertino, California. Apple's most 
        famous product is the iPhone, which revolutionized the smartphone industry. The company also makes 
        computers, tablets, and wearable devices. Apple's CEO is Tim Cook, who took over the role after 
        Steve Jobs passed away in 2011.
        """,
        """
        The research team published their findings in a prestigious journal. They had been working on this 
        project for three years. The lead researcher, Dr. Smith, explained that the results exceeded their 
        expectations. She mentioned that the methodology they used was innovative and could be applied to 
        other studies. The team plans to continue their research in this area.
        """
    ]
    
    for i, text in enumerate(complex_texts, 1):
        print(f"\n复杂示例 {i}:")
        print(f"原文: {text.strip()}")
        
        # 使用基础方法
        basic_resolved = resolver.resolve_coreferences(text.strip())
        print(f"\n基础解析: {basic_resolved}")
        
        # 使用改进方法
        improved_resolved = resolver.improved_replace_corefs(text.strip())
        print(f"\n改进解析: {improved_resolved}")


def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 60)
    print("交互式指代消解演示")
    print("=" * 60)
    print("输入文本进行指代消解分析（输入 'quit' 退出）")
    
    resolver = CoreferenceResolver()
    
    while True:
        text = input("\n请输入文本: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        try:
            analysis = resolver.analyze_coreferences(text)
            
            print(f"\n分析结果:")
            print(f"聚类数量: {analysis['num_clusters']}")
            
            if analysis['num_clusters'] > 0:
                for cluster in analysis['clusters']:
                    print(f"\n聚类 {cluster['cluster_id']}:")
                    print(f"  代表mention: {cluster['representative']['text']}")
                    print(f"  所有mentions:")
                    for mention in cluster['mentions']:
                        status = " (代表)" if mention['is_representative'] else ""
                        print(f"    - {mention['text']}{status}")
                
                print(f"\n解析后文本: {analysis['resolved_text']}")
            else:
                print("未发现指代关系")
                
        except Exception as e:
            print(f"处理出错: {e}")


def benchmark_example():
    """性能测试示例"""
    print("\n" + "=" * 60)
    print("性能测试示例")
    print("=" * 60)
    
    import time
    
    resolver = CoreferenceResolver()
    
    # 测试文本
    test_text = """
    The technology company announced its quarterly results yesterday. The company's revenue exceeded expectations, 
    driven by strong sales of its flagship products. CEO John Smith said he was pleased with the performance. 
    Smith noted that the company's investment in research and development was paying off. The firm's stock price 
    rose 5% in after-hours trading. Analysts said they were optimistic about the company's future prospects.
    """
    
    # 预热
    resolver.predict(test_text)
    
    # 性能测试
    num_runs = 10
    start_time = time.time()
    
    for _ in range(num_runs):
        resolver.resolve_coreferences(test_text)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    print(f"平均处理时间: {avg_time:.3f} 秒")
    print(f"处理 {num_runs} 次的总时间: {end_time - start_time:.3f} 秒")


def main():
    """主函数"""
    print("AllenNLP 指代消解示例")
    print("这个示例展示了如何使用AllenNLP进行指代消解")
    print("包括基础用法、高级技术和性能测试")
    
    try:
        # 运行基础示例
        basic_example()
        
        # 运行高级示例
        advanced_example()
        
        # 运行性能测试
        benchmark_example()
        
        # 交互式演示（可选）
        print(f"\n是否运行交互式演示？(y/n): ", end="")
        choice = input().strip().lower()
        if choice == 'y':
            interactive_demo()
        
        print("\n示例运行完成！")
        
    except Exception as e:
        print(f"运行出错: {e}")
        print("请确保已安装所有必要的依赖包")


if __name__ == "__main__":
    main()