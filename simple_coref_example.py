#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的AllenNLP指代消解示例
=========================

这是一个快速上手的指代消解示例，展示了核心功能。

使用方法:
1. 安装依赖: pip install -r requirements.txt
2. 下载spaCy模型: python -m spacy download en_core_web_sm
3. 运行示例: python simple_coref_example.py
"""

import warnings
warnings.filterwarnings("ignore")

def install_dependencies():
    """安装必要的依赖包"""
    import subprocess
    import sys
    
    packages = [
        "allennlp",
        "allennlp-models", 
        "spacy",
        "torch"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # 下载spaCy模型
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        print("正在下载spaCy英文模型...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

class SimpleCoreferenceResolver:
    """简单的指代消解器"""
    
    def __init__(self):
        """初始化"""
        try:
            from allennlp.predictors import Predictor
        except ImportError:
            install_dependencies()
            from allennlp.predictors import Predictor
        
        print("正在加载指代消解模型...")
        
        # 使用预训练的指代消解模型
        model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
        self.predictor = Predictor.from_path(model_url)
        
        print("模型加载完成！")
    
    def resolve(self, text: str) -> dict:
        """
        解析文本中的指代关系
        
        Args:
            text: 输入文本
            
        Returns:
            包含原文、聚类和解析结果的字典
        """
        # 预测指代关系
        prediction = self.predictor.predict(document=text)
        
        clusters = prediction.get("clusters", [])
        document = prediction["document"]
        
        # 构建结果
        result = {
            "original_text": text,
            "tokens": document,
            "clusters": clusters,
            "cluster_texts": [],
            "resolved_text": ""
        }
        
        # 提取聚类文本
        for cluster in clusters:
            cluster_texts = []
            for span in cluster:
                start, end = span[0], span[1]
                span_text = " ".join(document[start:end+1])
                cluster_texts.append(span_text)
            result["cluster_texts"].append(cluster_texts)
        
        # 生成解析后的文本
        result["resolved_text"] = self._resolve_text(document, clusters)
        
        return result
    
    def _resolve_text(self, document, clusters):
        """生成解析后的文本"""
        resolved = document.copy()
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # 选择最长的mention作为代表
            representative = max(cluster, key=lambda x: x[1] - x[0])
            rep_text = " ".join(document[representative[0]:representative[1]+1])
            
            # 替换其他mention
            for mention in sorted(cluster, key=lambda x: x[0], reverse=True):
                if mention != representative:
                    start, end = mention[0], mention[1]
                    # 简单替换
                    for i in range(start, end + 1):
                        if i == start:
                            resolved[i] = rep_text
                        else:
                            resolved[i] = ""
        
        return " ".join(word for word in resolved if word.strip())

def demo():
    """演示函数"""
    print("=" * 50)
    print("AllenNLP 指代消解简单示例")
    print("=" * 50)
    
    # 创建指代消解器
    resolver = SimpleCoreferenceResolver()
    
    # 测试句子
    test_sentences = [
        "约翰是一名工程师。他在谷歌工作。约翰喜欢他的工作。",  # 混合中英文可能有问题，这里用英文
        "John is a software engineer. He works at Google. John loves his job.",
        "The cat sat on the mat. It was very comfortable.",
        "Mary bought a book. She read it last night. The book was interesting.",
        "Apple Inc. announced its new iPhone. The company expects it to be successful."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n示例 {i}:")
        print(f"原文: {sentence}")
        
        try:
            result = resolver.resolve(sentence)
            
            print(f"找到 {len(result['clusters'])} 个指代聚类:")
            
            for j, cluster_texts in enumerate(result['cluster_texts']):
                print(f"  聚类 {j+1}: {' → '.join(cluster_texts)}")
            
            print(f"解析后: {result['resolved_text']}")
            
        except Exception as e:
            print(f"处理出错: {e}")
        
        print("-" * 30)

def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 50)
    print("交互式指代消解")
    print("=" * 50)
    print("输入英文文本进行指代消解（输入 'quit' 退出）")
    
    resolver = SimpleCoreferenceResolver()
    
    while True:
        try:
            text = input("\n请输入文本: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not text:
                print("请输入有效文本")
                continue
            
            result = resolver.resolve(text)
            
            print(f"\n分析结果:")
            print(f"聚类数量: {len(result['clusters'])}")
            
            if result['clusters']:
                for i, cluster_texts in enumerate(result['cluster_texts']):
                    print(f"聚类 {i+1}: {' ↔ '.join(cluster_texts)}")
                print(f"\n解析后文本: {result['resolved_text']}")
            else:
                print("未发现指代关系")
                
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"处理出错: {e}")

if __name__ == "__main__":
    # 运行演示
    demo()
    
    # 询问是否进入交互模式
    while True:
        choice = input("\n是否进入交互模式？(y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_mode()
            break
        elif choice in ['n', 'no']:
            print("示例结束！")
            break
        else:
            print("请输入 y 或 n")