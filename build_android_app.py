"""
Android应用自动化打包脚本
支持Kivy + Buildozer方案
"""

import os
import sys
import subprocess
import platform

def print_step(message):
    """打印步骤信息"""
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70)

def check_environment():
    """检查环境"""
    print_step("检查环境")
    
    # 检查操作系统
    system = platform.system()
    print(f"操作系统: {system}")
    
    if system == "Windows":
        print("\n⚠️  警告：Android打包需要Linux环境")
        print("\n推荐方案：")
        print("  1. 使用WSL2 (Windows Subsystem for Linux)")
        print("  2. 使用虚拟机 (VirtualBox + Ubuntu)")
        print("  3. 使用云服务器 (阿里云/腾讯云)")
        print("\n或者使用Streamlit Web方案：")
        print("  python stock_app_streamlit.py")
        
        choice = input("\n是否继续检查其他依赖？(y/n): ").strip().lower()
        if choice != 'y':
            return False
    
    # 检查Python版本
    python_version = sys.version
    print(f"\nPython版本: {python_version}")
    
    # 检查buildozer
    try:
        result = subprocess.run(['buildozer', '--version'], 
                              capture_output=True, text=True)
        print(f"Buildozer: ✅ 已安装")
    except FileNotFoundError:
        print("Buildozer: ❌ 未安装")
        print("\n安装命令:")
        print("  pip install buildozer cython==0.29.33")
        return False
    
    # 检查Java
    try:
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True)
        print("Java: ✅ 已安装")
    except FileNotFoundError:
        print("Java: ❌ 未安装")
        print("\n安装命令 (Ubuntu):")
        print("  sudo apt install openjdk-11-jdk")
        return False
    
    return True

def check_files():
    """检查必要文件"""
    print_step("检查必要文件")
    
    required_files = [
        'stock_app_android.py',
        'buildozer.spec',
        'stock_live_prediction_APP.py',
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - 文件不存在")
            all_exist = False
    
    if os.path.exists('models'):
        pkl_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        if pkl_files:
            print(f"✅ models/ - 包含 {len(pkl_files)} 个模型文件")
        else:
            print("⚠️  models/ - 文件夹存在但没有模型文件")
            all_exist = False
    else:
        print("❌ models/ - 文件夹不存在")
        all_exist = False
    
    return all_exist

def install_dependencies():
    """安装依赖"""
    print_step("安装依赖")
    
    print("是否安装/更新打包依赖？")
    print("  1. 只安装buildozer")
    print("  2. 安装所有依赖（buildozer + kivy + 其他）")
    print("  3. 跳过")
    
    choice = input("\n选择 (1/2/3): ").strip()
    
    if choice == '1':
        print("\n安装buildozer...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'buildozer', 'cython==0.29.33'])
    elif choice == '2':
        print("\n安装所有依赖...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'buildozer', 'cython==0.29.33', 'kivy'])
    
    return True

def build_apk():
    """构建APK"""
    print_step("构建APK")
    
    print("选择构建模式：")
    print("  1. Debug模式（开发测试用）")
    print("  2. Release模式（发布用，需要签名）")
    print("  3. 清理后重新构建")
    
    choice = input("\n选择 (1/2/3): ").strip()
    
    if choice == '3':
        print("\n清理旧文件...")
        subprocess.run(['buildozer', 'android', 'clean'])
        print("\n重新构建...")
        choice = '1'
    
    if choice == '1':
        print("\n开始构建Debug APK...")
        print("⏱️  首次构建需要下载SDK/NDK，可能需要1-2小时")
        print("💡 后续构建只需5-10分钟\n")
        
        result = subprocess.run(['buildozer', '-v', 'android', 'debug'])
        
        if result.returncode == 0:
            print("\n✅ APK构建成功！")
            print("\n文件位置: bin/stockpredictor-1.0-debug.apk")
            return True
        else:
            print("\n❌ APK构建失败")
            return False
            
    elif choice == '2':
        print("\n开始构建Release APK...")
        result = subprocess.run(['buildozer', '-v', 'android', 'release'])
        
        if result.returncode == 0:
            print("\n✅ APK构建成功！")
            print("\n⚠️  Release APK需要签名才能安装")
            print("文件位置: bin/stockpredictor-1.0-release-unsigned.apk")
            return True
        else:
            print("\n❌ APK构建失败")
            return False
    
    return False

def test_apk():
    """测试APK"""
    print_step("测试APK")
    
    # 查找APK文件
    apk_file = None
    if os.path.exists('bin'):
        apk_files = [f for f in os.listdir('bin') if f.endswith('.apk')]
        if apk_files:
            apk_file = os.path.join('bin', apk_files[0])
    
    if not apk_file:
        print("❌ 未找到APK文件")
        return False
    
    print(f"找到APK: {apk_file}")
    
    # 检查文件大小
    size_mb = os.path.getsize(apk_file) / (1024 * 1024)
    print(f"文件大小: {size_mb:.2f} MB")
    
    print("\n安装选项：")
    print("  1. 安装到模拟器")
    print("  2. 安装到真机（需USB连接）")
    print("  3. 跳过安装")
    
    choice = input("\n选择 (1/2/3): ").strip()
    
    if choice in ['1', '2']:
        print("\n检查ADB连接...")
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        print(result.stdout)
        
        print(f"\n安装APK到设备...")
        result = subprocess.run(['adb', 'install', '-r', apk_file])
        
        if result.returncode == 0:
            print("\n✅ APK安装成功！")
            print("\n💡 提示：")
            print("  - 在设备上找到 '股票预测系统' 应用")
            print("  - 首次运行可能需要授予网络权限")
            print("  - 查看日志: adb logcat | grep python")
            return True
        else:
            print("\n❌ APK安装失败")
            return False
    
    return True

def show_streamlit_option():
    """显示Streamlit替代方案"""
    print_step("推荐：Streamlit Web方案")
    
    print("""
Streamlit Web方案的优势：
  ✅ 不需要Linux环境，Windows即可开发
  ✅ 完全兼容所有Python库（pandas, sklearn等）
  ✅ 开发速度快，1小时即可完成
  ✅ 跨平台（手机/电脑都能用）
  ✅ 易于更新和维护

使用方法：
  1. 安装Streamlit:
     pip install streamlit
  
  2. 运行应用:
     streamlit run stock_app_streamlit.py
  
  3. 在手机浏览器访问显示的URL
  
  4. 可选：部署到云端（Heroku/Railway/Streamlit Cloud）
     用户通过网址直接访问，无需安装
  
  5. 可选：用Capacitor/Cordova封装成原生APP
     获得完整的APP体验

是否现在启动Streamlit应用？(y/n): """)
    
    choice = input().strip().lower()
    
    if choice == 'y':
        if not os.path.exists('stock_app_streamlit.py'):
            print("❌ stock_app_streamlit.py 文件不存在")
            return
        
        print("\n启动Streamlit应用...")
        print("💡 按 Ctrl+C 可以停止\n")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'stock_app_streamlit.py'])

def main():
    """主函数"""
    print("="*70)
    print("  📱 股票预测系统 - Android打包工具")
    print("="*70)
    
    # 检查环境
    if not check_environment():
        print("\n环境检查未通过")
        show_streamlit_option()
        return
    
    # 检查文件
    if not check_files():
        print("\n❌ 必要文件缺失，无法继续")
        return
    
    # 安装依赖
    install_dependencies()
    
    # 构建APK
    if build_apk():
        # 测试APK
        test_apk()
    else:
        print("\n构建失败，请检查错误信息")
        print("\n常见问题：")
        print("  1. 首次构建需要下载大量文件，确保网络连接稳定")
        print("  2. 某些Python包在Android上编译困难（如pandas）")
        print("  3. 需要足够的磁盘空间（至少10GB）")
        print("\n建议：使用Streamlit Web方案")
        
        show_streamlit_option()
    
    print("\n" + "="*70)
    print("  完成！")
    print("="*70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


