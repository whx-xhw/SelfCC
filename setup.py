#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AllenNLP指代消解示例安装脚本
============================

这个脚本会自动安装所有必要的依赖并下载模型文件。

使用方法:
python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*50}")
    print(f"正在{description}...")
    print(f"{'='*50}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=True, 
                                  capture_output=True, text=True)
        
        print(f"✅ {description}完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}失败:")
        print(f"错误: {e}")
        if e.stdout:
            print(f"输出: {e.stdout}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("请使用Python 3.7或更高版本")
        return False
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def install_basic_packages():
    """安装基础包"""
    basic_packages = [
        "pip>=21.0",
        "wheel",
        "setuptools>=50.0"
    ]
    
    for package in basic_packages:
        if not run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", package],
            f"安装{package}"
        ):
            return False
    
    return True

def install_pytorch():
    """安装PyTorch"""
    print("\n检查PyTorch安装...")
    
    try:
        import torch
        print(f"✅ PyTorch已安装: {torch.__version__}")
        return True
    except ImportError:
        pass
    
    # 安装CPU版本的PyTorch
    return run_command(
        [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"],
        "安装PyTorch"
    )

def install_main_packages():
    """安装主要包"""
    packages = [
        "allennlp>=2.10.0",
        "allennlp-models>=2.10.0", 
        "spacy>=3.4.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "requests>=2.28.0"
    ]
    
    for package in packages:
        if not run_command(
            [sys.executable, "-m", "pip", "install", package],
            f"安装{package}"
        ):
            return False
    
    return True

def download_spacy_models():
    """下载spaCy模型"""
    models = [
        ("en_core_web_sm", "英文模型"),
        # ("zh_core_web_sm", "中文模型")  # 可选
    ]
    
    for model, description in models:
        if not run_command(
            [sys.executable, "-m", "spacy", "download", model],
            f"下载spaCy {description}"
        ):
            print(f"⚠️  {description}下载失败，但程序仍可运行")
    
    return True

def test_installation():
    """测试安装是否成功"""
    print(f"\n{'='*50}")
    print("测试安装...")
    print(f"{'='*50}")
    
    test_code = '''
import sys
try:
    import allennlp
    print(f"✅ AllenNLP: {allennlp.__version__}")
except ImportError as e:
    print(f"❌ AllenNLP导入失败: {e}")
    sys.exit(1)

try:
    import allennlp_models
    print("✅ AllenNLP Models: 可用")
except ImportError as e:
    print(f"❌ AllenNLP Models导入失败: {e}")
    sys.exit(1)

try:
    import spacy
    print(f"✅ spaCy: {spacy.__version__}")
except ImportError as e:
    print(f"❌ spaCy导入失败: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch导入失败: {e}")
    sys.exit(1)

try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy英文模型: 可用")
except OSError as e:
    print(f"❌ spaCy英文模型加载失败: {e}")
    sys.exit(1)

print("\\n🎉 所有组件安装成功！")
'''
    
    return run_command(
        [sys.executable, "-c", test_code],
        "验证安装"
    )

def create_test_file():
    """创建测试文件"""
    test_code = '''#!/usr/bin/env python3
"""
快速测试AllenNLP指代消解
"""

def quick_test():
    try:
        from allennlp.predictors import Predictor
        
        print("正在加载模型...")
        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
        )
        
        test_text = "John is an engineer. He works at Google."
        print(f"测试文本: {test_text}")
        
        result = predictor.predict(document=test_text)
        clusters = result.get("clusters", [])
        
        print(f"找到 {len(clusters)} 个指代聚类")
        if clusters:
            for i, cluster in enumerate(clusters):
                mentions = []
                for span in cluster:
                    mention = " ".join(result["document"][span[0]:span[1]+1])
                    mentions.append(mention)
                print(f"聚类 {i+1}: {' → '.join(mentions)}")
        
        print("\\n✅ 测试成功！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    quick_test()
'''
    
    with open("test_installation.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    
    print("✅ 创建了测试文件: test_installation.py")

def main():
    """主安装流程"""
    print("🚀 AllenNLP指代消解示例安装程序")
    print("="*60)
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 安装基础包
    if not install_basic_packages():
        print("❌ 基础包安装失败")
        return False
    
    # 安装PyTorch
    if not install_pytorch():
        print("❌ PyTorch安装失败")
        return False
    
    # 安装主要包
    if not install_main_packages():
        print("❌ 主要包安装失败")
        return False
    
    # 下载spaCy模型
    download_spacy_models()
    
    # 测试安装
    if not test_installation():
        print("❌ 安装验证失败")
        return False
    
    # 创建测试文件
    create_test_file()
    
    print(f"\n{'='*60}")
    print("🎉 安装完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 运行简单示例: python simple_coref_example.py")
    print("2. 运行完整示例: python allennlp_coreference_example.py") 
    print("3. 快速测试: python test_installation.py")
    print("\n如有问题，请查看README.md或检查网络连接。")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)