# -*- coding: utf-8 -*-
"""
修复Streamlit权限错误问题
诊断和解决PermissionError
"""

import os
import sys
import pickle
import subprocess

def check_file_permissions():
    """检查关键文件的权限"""
    print("="*70)
    print("[检查] 文件权限诊断")
    print("="*70)
    
    critical_files = [
        'models/trained_model.pkl',
        'models/all_trained_models.pkl',
        'models/feature_list.pkl',
        'models/model_info.pkl',
        'data/all_stock_data.pkl',
    ]
    
    issues_found = []
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            try:
                # 尝试读取文件
                with open(file_path, 'rb') as f:
                    f.read(100)  # 只读取开头测试
                
                # 检查文件大小
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                # 检查是否可写
                writable = os.access(file_path, os.W_OK)
                
                status = "[OK]" if writable else "[只读]"
                print(f"{status} {file_path} ({size_mb:.2f} MB)")
                
                if not writable:
                    issues_found.append(f"{file_path} - 只读，可能导致问题")
                    
            except PermissionError as e:
                print(f"[错误] {file_path} - 权限错误: {e}")
                issues_found.append(f"{file_path} - 权限错误")
            except Exception as e:
                print(f"[错误] {file_path} - {e}")
                issues_found.append(f"{file_path} - {e}")
        else:
            print(f"[缺失] {file_path} - 文件不存在")
            issues_found.append(f"{file_path} - 文件不存在")
    
    return issues_found

def check_streamlit_cache():
    """检查Streamlit缓存"""
    print("\n" + "="*70)
    print("[检查] Streamlit缓存")
    print("="*70)
    
    cache_dir = os.path.expanduser('~/.streamlit')
    
    if os.path.exists(cache_dir):
        print(f"[找到] 缓存目录: {cache_dir}")
        
        # 清空缓存
        choice = input("\n是否清空Streamlit缓存？(y/n): ").strip().lower()
        if choice == 'y':
            try:
                import shutil
                shutil.rmtree(cache_dir)
                print("[完成] 缓存已清空")
                return True
            except Exception as e:
                print(f"[错误] 清空缓存失败: {e}")
                return False
    else:
        print("[信息] 缓存目录不存在")
    
    return True

def check_running_processes():
    """检查是否有其他Streamlit进程正在运行"""
    print("\n" + "="*70)
    print("[检查] 运行中的Streamlit进程")
    print("="*70)
    
    try:
        # Windows
        result = subprocess.run(
            'tasklist | findstr streamlit',
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            print("[警告] 发现运行中的Streamlit进程：")
            print(result.stdout)
            print("\n请先关闭这些进程，或按Ctrl+C终止")
            
            choice = input("\n是否尝试终止所有Streamlit进程？(y/n): ").strip().lower()
            if choice == 'y':
                subprocess.run('taskkill /F /IM streamlit.exe', shell=True)
                print("[完成] 已尝试终止进程")
        else:
            print("[OK] 没有发现运行中的Streamlit进程")
            
    except Exception as e:
        print(f"[信息] 无法检查进程: {e}")

def test_model_loading():
    """测试模型文件加载"""
    print("\n" + "="*70)
    print("[测试] 模型文件加载")
    print("="*70)
    
    model_files = {
        'trained_model.pkl': 'models/trained_model.pkl',
        'all_trained_models.pkl': 'models/all_trained_models.pkl',
        'feature_list.pkl': 'models/feature_list.pkl',
        'model_info.pkl': 'models/model_info.pkl',
    }
    
    failed = []
    
    for name, path in model_files.items():
        if not os.path.exists(path):
            print(f"[跳过] {name} - 文件不存在")
            continue
            
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"[成功] {name} - 加载成功")
        except PermissionError as e:
            print(f"[失败] {name} - 权限错误: {e}")
            failed.append(name)
        except Exception as e:
            print(f"[失败] {name} - {e}")
            failed.append(name)
    
    return failed

def create_streamlit_config():
    """创建Streamlit配置文件"""
    print("\n" + "="*70)
    print("[配置] Streamlit设置")
    print("="*70)
    
    config_dir = '.streamlit'
    config_file = os.path.join(config_dir, 'config.toml')
    
    # 创建目录
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"[创建] 配置目录: {config_dir}")
    
    # 创建配置文件
    config_content = """[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[runner]
magicEnabled = false
installTracer = false
fixMatplotlib = true

[client]
showErrorDetails = true
"""
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"[完成] 配置文件已创建: {config_file}")
        return True
    except Exception as e:
        print(f"[错误] 创建配置失败: {e}")
        return False

def fix_file_permissions():
    """尝试修复文件权限"""
    print("\n" + "="*70)
    print("[修复] 文件权限")
    print("="*70)
    
    choice = input("是否尝试修复文件权限？(y/n): ").strip().lower()
    
    if choice != 'y':
        return
    
    # 在Windows上，尝试修改文件属性
    critical_dirs = ['models', 'data']
    
    for dir_name in critical_dirs:
        if os.path.exists(dir_name):
            try:
                # 移除只读属性
                subprocess.run(f'attrib -R {dir_name}\\*.* /S', shell=True, check=False)
                print(f"[完成] {dir_name}/ - 已移除只读属性")
            except Exception as e:
                print(f"[错误] {dir_name}/ - {e}")

def suggest_solutions(issues):
    """根据问题提供解决方案"""
    print("\n" + "="*70)
    print("[建议] 解决方案")
    print("="*70)
    
    if not issues:
        print("[OK] 未发现问题！")
        print("\n您可以尝试运行:")
        print("  streamlit run stock_app_streamlit.py")
        return
    
    print("\n发现以下问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n[推荐解决方案]")
    print("1. 关闭所有可能占用文件的程序（VS Code、Jupyter等）")
    print("2. 清空Streamlit缓存")
    print("3. 移除文件只读属性")
    print("4. 如果问题仍然存在，重新训练模型")
    print("\n[命令]")
    print("  清空缓存: streamlit cache clear")
    print("  移除只读: attrib -R models\\*.* /S")

def main():
    """主函数"""
    print("="*70)
    print("  Streamlit权限错误诊断和修复工具")
    print("="*70)
    print()
    print("此工具将帮助您诊断和修复Streamlit的PermissionError")
    print()
    
    # 1. 检查文件权限
    issues = check_file_permissions()
    
    # 2. 检查运行中的进程
    check_running_processes()
    
    # 3. 测试模型加载
    failed_models = test_model_loading()
    if failed_models:
        issues.extend([f"模型加载失败: {m}" for m in failed_models])
    
    # 4. 检查Streamlit缓存
    check_streamlit_cache()
    
    # 5. 创建Streamlit配置
    create_streamlit_config()
    
    # 6. 修复文件权限
    fix_file_permissions()
    
    # 7. 提供建议
    suggest_solutions(issues)
    
    print("\n" + "="*70)
    print("[完成] 诊断结束")
    print("="*70)
    print()
    print("现在可以尝试重新运行Streamlit:")
    print("  streamlit run stock_app_streamlit.py")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[中断] 操作被用户中断")
    except Exception as e:
        print(f"\n[错误] 发生错误: {e}")
        import traceback
        traceback.print_exc()

