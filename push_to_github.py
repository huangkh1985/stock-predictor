"""
GitHub推送自动化脚本
自动完成Git初始化、添加、提交、推送
"""

import subprocess
import sys
import os

def print_step(step_num, message):
    """打印步骤信息"""
    print("\n" + "="*70)
    print(f"步骤 {step_num}: {message}")
    print("="*70)

def run_command(command, error_message, check=True):
    """运行命令"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        if result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if check:
            print(f"❌ {error_message}")
            if e.stderr:
                print(f"错误信息: {e.stderr}")
        return False, e.stderr

def check_git():
    """检查Git是否安装"""
    print_step(1, "检查Git安装")
    
    success, output = run_command("git --version", "Git未安装", check=False)
    
    if success:
        print(f"✅ Git已安装: {output.strip()}")
        return True
    else:
        print("❌ Git未安装")
        print("\n请访问 https://git-scm.com/download/win 下载安装")
        print("或使用命令安装: choco install git")
        return False

def check_git_config():
    """检查Git配置"""
    print_step(2, "检查Git配置")
    
    success, name = run_command("git config --global user.name", "", check=False)
    success2, email = run_command("git config --global user.email", "", check=False)
    
    if name.strip() and email.strip():
        print(f"✅ 用户名: {name.strip()}")
        print(f"✅ 邮箱: {email.strip()}")
        return True
    else:
        print("⚠️  Git未配置用户信息")
        print("\n请输入配置信息：")
        
        username = input("GitHub用户名: ").strip()
        email_input = input("GitHub邮箱: ").strip()
        
        if username and email_input:
            run_command(f'git config --global user.name "{username}"', "配置失败")
            run_command(f'git config --global user.email "{email_input}"', "配置失败")
            print("✅ 配置成功")
            return True
        else:
            print("❌ 配置信息不能为空")
            return False

def create_gitignore():
    """创建.gitignore文件"""
    print_step(3, "创建.gitignore文件")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp

# 数据文件
*.csv
*.xlsx
*.db
*.sqlite

# 图片（可选）
# *.png
# *.jpg

# 日志
*.log

# 临时文件
*.tmp
.DS_Store
Thumbs.db

# 打包文件
dist/
build/
bin/
.buildozer/

# 其他
.env
*.bak
"""
    
    if os.path.exists('.gitignore'):
        print("⚠️  .gitignore已存在")
        choice = input("是否覆盖？(y/n): ").strip().lower()
        if choice != 'y':
            print("跳过创建.gitignore")
            return True
    
    try:
        with open('.gitignore', 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("✅ .gitignore创建成功")
        return True
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return False

def create_requirements():
    """创建requirements.txt"""
    print_step(4, "创建requirements.txt")
    
    requirements_content = """streamlit>=1.20.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tsfresh>=0.19.0
efinance>=0.3.0
lightgbm>=3.3.0
xgboost>=1.5.0
joblib>=1.0.0
python-dateutil>=2.8.0
"""
    
    if os.path.exists('requirements.txt'):
        print("⚠️  requirements.txt已存在")
        choice = input("是否覆盖？(y/n): ").strip().lower()
        if choice != 'y':
            print("保留现有requirements.txt")
            return True
    
    try:
        with open('requirements.txt', 'w', encoding='utf-8') as f:
            f.write(requirements_content)
        print("✅ requirements.txt创建成功")
        return True
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return False

def check_required_files():
    """检查必要文件"""
    print_step(5, "检查必要文件")
    
    required_files = [
        'stock_app_streamlit.py',
        'stock_live_prediction_APP.py',
    ]
    
    optional_files = [
        'requirements.txt',
        '.gitignore',
    ]
    
    all_ok = True
    
    print("\n必需文件:")
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - 文件不存在")
            all_ok = False
    
    print("\n可选文件:")
    for file in optional_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} - 建议创建")
    
    if os.path.exists('models'):
        pkl_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        if pkl_files:
            print(f"\n模型文件: ✅ 找到 {len(pkl_files)} 个模型文件")
            
            # 检查大小
            for pkl in pkl_files[:3]:  # 只显示前3个
                size = os.path.getsize(os.path.join('models', pkl)) / (1024*1024)
                if size > 100:
                    print(f"  ⚠️  {pkl}: {size:.2f}MB (超过100MB，需要Git LFS)")
                else:
                    print(f"  ✅ {pkl}: {size:.2f}MB")
        else:
            print("\n模型文件: ⚠️  models/文件夹存在但没有.pkl文件")
    else:
        print("\n模型文件: ❌ models/文件夹不存在")
        all_ok = False
    
    return all_ok

def git_init():
    """初始化Git仓库"""
    print_step(6, "初始化Git仓库")
    
    if os.path.exists('.git'):
        print("⚠️  Git仓库已存在")
        choice = input("是否重新初始化？(y/n): ").strip().lower()
        if choice == 'y':
            print("\n正在删除旧的Git仓库...")
            # Windows上使用更强制的删除方法
            if sys.platform == 'win32':
                # 先尝试使用Windows命令强制删除
                result = subprocess.run(
                    'rmdir /s /q .git',
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print("⚠️  无法自动删除.git文件夹")
                    print("\n请手动操作：")
                    print("  1. 关闭所有可能占用文件的程序（如VS Code、Git GUI等）")
                    print("  2. 打开PowerShell，运行：")
                    print("     Remove-Item -Recurse -Force .git")
                    print("  3. 或者使用文件管理器手动删除.git文件夹")
                    print("\n删除后请重新运行此脚本")
                    return False
            else:
                # Linux/Mac使用shutil
                import shutil
                try:
                    shutil.rmtree('.git')
                except Exception as e:
                    print(f"❌ 删除失败: {e}")
                    print("\n请手动删除.git文件夹后重新运行")
                    return False
            
            print("✅ 旧仓库已删除")
        else:
            print("使用现有Git仓库")
            # 验证Git仓库是否有效
            success, _ = run_command("git status", "", check=False)
            if not success:
                print("\n❌ 现有Git仓库已损坏！")
                print("建议选择'y'重新初始化")
                return False
            return True
    
    success, _ = run_command("git init", "初始化失败")
    
    if success:
        print("✅ Git仓库初始化成功")
        # 创建main分支
        run_command("git branch -M main", "创建分支失败", check=False)
        return True
    return False

def git_add():
    """添加文件到暂存区"""
    print_step(7, "添加文件到Git")
    
    print("\n添加所有文件...")
    success, _ = run_command("git add .", "添加文件失败")
    
    if success:
        print("\n查看将要提交的文件:")
        run_command("git status", "查看状态失败", check=False)
        
        print("\n⚠️  请检查上面的文件列表")
        choice = input("确认要提交这些文件吗？(y/n): ").strip().lower()
        
        if choice == 'y':
            print("✅ 文件已添加")
            return True
        else:
            print("❌ 取消添加")
            run_command("git reset", "重置失败", check=False)
            return False
    
    return False

def git_commit():
    """提交到本地仓库"""
    print_step(8, "提交到本地仓库")
    
    print("\n请输入提交说明（留空使用默认）:")
    commit_msg = input("提交说明: ").strip()
    
    if not commit_msg:
        commit_msg = "Initial commit: stock prediction app"
    
    success, _ = run_command(f'git commit -m "{commit_msg}"', "提交失败")
    
    if success:
        print("✅ 提交成功")
        return True
    else:
        print("⚠️  可能没有要提交的更改")
        return False

def git_remote():
    """配置远程仓库"""
    print_step(9, "配置GitHub远程仓库")
    
    # 检查是否已有远程仓库
    success, output = run_command("git remote -v", "", check=False)
    
    if success and output.strip():
        print("\n当前远程仓库:")
        print(output)
        choice = input("\n是否重新配置？(y/n): ").strip().lower()
        if choice != 'y':
            return True
        else:
            run_command("git remote remove origin", "", check=False)
    
    print("\n请输入GitHub仓库信息:")
    print("格式: https://github.com/用户名/仓库名.git")
    print("示例: https://github.com/zhangsan/stock-predictor.git")
    print()
    
    repo_url = input("GitHub仓库URL: ").strip()
    
    if not repo_url:
        print("❌ 仓库URL不能为空")
        print("\n请先在GitHub创建仓库:")
        print("  1. 访问 https://github.com/new")
        print("  2. 创建仓库（不要勾选任何初始化选项）")
        print("  3. 复制仓库URL")
        return False
    
    success, _ = run_command(f"git remote add origin {repo_url}", "添加远程仓库失败")
    
    if success:
        print("✅ 远程仓库配置成功")
        return True
    return False

def git_push():
    """推送到GitHub"""
    print_step(10, "推送到GitHub")
    
    print("\n⚠️  首次推送需要登录GitHub")
    print("提示:")
    print("  - 用户名: 你的GitHub用户名")
    print("  - 密码: 使用Personal Access Token（不是密码）")
    print()
    print("如何获取Token:")
    print("  1. 访问 https://github.com/settings/tokens")
    print("  2. Generate new token (classic)")
    print("  3. 勾选 'repo'")
    print("  4. 复制Token")
    print()
    
    choice = input("准备好推送了吗？(y/n): ").strip().lower()
    
    if choice != 'y':
        print("取消推送")
        return False
    
    print("\n开始推送...")
    success, output = run_command("git push -u origin main", "推送失败", check=False)
    
    if success or "Everything up-to-date" in output:
        print("\n✅ 推送成功！")
        print("\n请在浏览器访问你的GitHub仓库查看")
        return True
    else:
        print("\n❌ 推送失败")
        print("\n常见问题:")
        print("  1. 凭据错误: 使用Personal Access Token而不是密码")
        print("  2. 权限不足: 检查仓库URL是否正确")
        print("  3. 网络问题: 检查网络连接")
        return False

def main():
    """主函数"""
    print("="*70)
    print("  📦 GitHub推送自动化工具")
    print("="*70)
    print()
    print("此工具将帮助你:")
    print("  1. 初始化Git仓库")
    print("  2. 创建必要文件（.gitignore, requirements.txt）")
    print("  3. 提交代码")
    print("  4. 推送到GitHub")
    print()
    
    choice = input("是否继续？(y/n): ").strip().lower()
    if choice != 'y':
        print("取消操作")
        return
    
    # 执行步骤
    steps = [
        (check_git, "检查Git"),
        (check_git_config, "配置Git"),
        (create_gitignore, "创建.gitignore"),
        (create_requirements, "创建requirements.txt"),
        (check_required_files, "检查文件"),
        (git_init, "初始化Git"),
        (git_add, "添加文件"),
        (git_commit, "提交"),
        (git_remote, "配置远程"),
        (git_push, "推送"),
    ]
    
    for step_func, step_name in steps:
        if not step_func():
            print(f"\n❌ {step_name}失败，推送中止")
            print("\n请解决问题后重新运行此脚本")
            return
    
    print("\n" + "="*70)
    print("  ✅ 推送完成！")
    print("="*70)
    print()
    print("🎉 代码已成功推送到GitHub！")
    print()
    print("📋 下一步:")
    print("  1. 访问你的GitHub仓库检查文件")
    print("  2. 部署到Streamlit Cloud:")
    print("     - 访问 https://share.streamlit.io")
    print("     - 选择你的仓库")
    print("     - 点击Deploy")
    print()
    print("  3. 查看部署教程: Streamlit_Cloud部署教程.md")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

