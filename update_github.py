"""
GitHub仓库更新脚本
快速添加、提交、推送更改到GitHub
"""

import subprocess
import sys
import os

def run_command(command, show_output=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        if show_output and result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if show_output:
            print(f"[错误] 命令执行失败: {command}")
            if e.stderr:
                print(f"错误信息: {e.stderr}")
        return False, e.stderr

def check_status():
    """检查当前状态"""
    print("\n" + "="*70)
    print("[状态] 当前Git状态")
    print("="*70)
    
    # 检查是否有修改
    success, output = run_command("git status --short", show_output=False)
    
    if success:
        if output.strip():
            print("\n[发现更改] 以下文件有变动：")
            print(output)
            return True
        else:
            print("\n[完成] 工作区干净，没有需要提交的更改")
            return False
    return False

def show_changes():
    """显示详细更改"""
    print("\n" + "="*70)
    print("[详细] 更改信息")
    print("="*70)
    run_command("git status")

def quick_update():
    """快速更新（一键完成）"""
    print("\n" + "="*70)
    print("[快速模式] 自动更新")
    print("="*70)
    
    print("\n请输入提交说明（留空使用默认）:")
    commit_msg = input("提交说明: ").strip()
    
    if not commit_msg:
        commit_msg = "Update: 更新项目文件"
    
    print("\n开始执行更新...")
    
    # 1. 添加所有文件
    print("\n[1/3] 添加文件...")
    success, _ = run_command("git add .", show_output=False)
    if not success:
        print("[失败] 添加文件失败")
        return False
    print("[完成] 文件已添加")
    
    # 2. 提交
    print("\n[2/3] 提交更改...")
    success, output = run_command(f'git commit -m "{commit_msg}"', show_output=False)
    if not success:
        if "nothing to commit" in str(output):
            print("[提示] 没有需要提交的更改")
        else:
            print("[失败] 提交失败")
        return False
    print("[完成] 提交成功")
    
    # 3. 推送
    print("\n[3/3] 推送到GitHub...")
    success, output = run_command("git push -u origin main", show_output=False)
    if success or "Everything up-to-date" in str(output):
        print("[完成] 推送成功")
        print("\n" + "="*70)
        print("[成功] 更新完成！代码已上传到GitHub")
        print("="*70)
        return True
    else:
        print("[失败] 推送失败")
        if output:
            print(output)
        
        # 如果是因为远程仓库不存在
        if "Repository not found" in str(output):
            print("\n[提示] 可能的原因：")
            print("  1. 远程仓库不存在 - 请先在GitHub创建仓库")
            print("  2. 仓库URL错误 - 检查仓库地址是否正确")
            print("  3. 没有访问权限 - 检查是否已登录GitHub")
            
            print("\n是否更新远程仓库地址？(y/n): ", end='')
            choice = input().strip().lower()
            if choice == 'y':
                repo_url = input("新的GitHub仓库URL: ").strip()
                if repo_url:
                    run_command("git remote remove origin", show_output=False)
                    run_command(f"git remote add origin {repo_url}")
                    print("\n再次尝试推送...")
                    success, output = run_command("git push -u origin main", show_output=False)
                    if success:
                        print("[完成] 推送成功！")
                        return True
        
        return False

def main():
    """主函数"""
    print("="*70)
    print("  GitHub仓库更新工具")
    print("="*70)
    print()
    print("此工具帮助您快速更新GitHub仓库")
    print()
    
    # 检查当前状态
    has_changes = check_status()
    
    if not has_changes:
        print("\n[提示] 没有需要更新的内容")
        
        # 显示最近的提交
        print("\n[信息] 最近的提交记录：")
        run_command("git log --oneline -5", show_output=True)
        
        print("\n[信息] 远程仓库：")
        run_command("git remote -v", show_output=True)
        
        return
    
    print("\n" + "="*70)
    print("选择更新模式:")
    print("="*70)
    print("  1. [快速] 一键完成（添加→提交→推送）")
    print("  2. [查看] 只查看更改（不提交）")
    print("  0. [退出]")
    print()
    
    choice = input("请选择 [1/2/0]: ").strip()
    
    if choice == '1':
        # 快速模式
        quick_update()
        
    elif choice == '2':
        # 只查看
        show_changes()
        
    elif choice == '0':
        print("\n[退出] 再见！")
        return
    
    else:
        print("\n[错误] 无效选择")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[中断] 操作被用户中断")
    except Exception as e:
        print(f"\n[错误] 发生错误: {e}")
        import traceback
        traceback.print_exc()
