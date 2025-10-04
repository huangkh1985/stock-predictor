"""
Streamlit智能启动脚本
自动识别正确的IP地址，避免虚拟网卡干扰
"""

import socket
import subprocess
import sys
import platform

def get_local_ip():
    """获取本机真实IP地址（排除虚拟网卡）"""
    
    ip_list = []
    
    try:
        # 方法1：通过连接外部地址获取（最可靠）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        main_ip = s.getsockname()[0]
        s.close()
        ip_list.append(("主要IP（推荐）", main_ip))
    except:
        pass
    
    # 方法2：获取所有网络接口
    try:
        hostname = socket.gethostname()
        addresses = socket.getaddrinfo(hostname, None)
        
        for addr in addresses:
            if addr[0] == socket.AF_INET:  # IPv4
                ip = addr[4][0]
                
                # 过滤掉不合适的IP
                if ip.startswith('127.'):  # 回环地址
                    continue
                elif ip.startswith('169.254.'):  # APIPA地址
                    continue
                elif ip.startswith('198.18.'):  # 虚拟网卡
                    continue
                elif ip in [x[1] for x in ip_list]:  # 已存在
                    continue
                
                # 判断IP类型
                if ip.startswith('192.168.'):
                    ip_list.append(("WiFi/有线局域网", ip))
                elif ip.startswith('10.'):
                    ip_list.append(("局域网", ip))
                elif ip.startswith('172.'):
                    third_octet = int(ip.split('.')[1])
                    if 16 <= third_octet <= 31:
                        ip_list.append(("局域网", ip))
                else:
                    ip_list.append(("其他网络", ip))
    except:
        pass
    
    return ip_list

def show_network_info():
    """显示网络信息"""
    print("="*70)
    print("  🌐 网络配置检测")
    print("="*70)
    
    ip_list = get_local_ip()
    
    if not ip_list:
        print("\n❌ 无法获取网络IP地址")
        print("\n可能原因：")
        print("  1. 未连接到网络")
        print("  2. 网络适配器被禁用")
        print("  3. 防火墙阻止了网络检测")
        print("\n建议：")
        print("  1. 检查网络连接")
        print("  2. 手动运行: streamlit run stock_app_streamlit.py")
        return None
    
    print(f"\n✅ 检测到 {len(ip_list)} 个可用IP地址：")
    print()
    
    for i, (network_type, ip) in enumerate(ip_list, 1):
        print(f"  {i}. [{network_type}] {ip}")
        if i == 1:
            print(f"     👉 推荐使用这个IP")
    
    # 选择推荐的IP（通常是第一个）
    recommended_ip = ip_list[0][1]
    
    print("\n" + "="*70)
    print("  📱 手机访问地址")
    print("="*70)
    print(f"\n  在手机浏览器输入：")
    print(f"  👉 http://{recommended_ip}:8501")
    print()
    
    if len(ip_list) > 1:
        print("  如果上面的地址无法访问，尝试：")
        for i, (network_type, ip) in enumerate(ip_list[1:], 2):
            print(f"     {i}. http://{ip}:8501")
        print()
    
    print("="*70)
    print("  ⚠️  重要提示")
    print("="*70)
    print("\n  1. 确保手机和电脑连接到同一WiFi网络")
    print("  2. 如果无法访问，请临时关闭防火墙测试")
    print("  3. 某些路由器可能阻止设备间通信（AP隔离）")
    print()
    
    return recommended_ip

def check_streamlit():
    """检查Streamlit是否安装"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def check_app_file():
    """检查应用文件是否存在"""
    import os
    
    files_to_check = [
        'stock_app_streamlit.py',
        'stock_live_prediction_APP.py',
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            return file
    
    return None

def main():
    """主函数"""
    print("="*70)
    print("  📊 股票预测系统 - Streamlit启动器")
    print("="*70)
    print()
    
    # 检查Streamlit
    print("⏳ 检查依赖...")
    if not check_streamlit():
        print("❌ Streamlit未安装")
        print("\n正在安装Streamlit...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'streamlit'])
        print()
    else:
        print("✅ Streamlit已安装")
    
    # 检查应用文件
    app_file = check_app_file()
    if not app_file:
        print("\n❌ 找不到应用文件")
        print("请确保以下文件之一存在：")
        print("  - stock_app_streamlit.py")
        print("  - stock_live_prediction_APP.py")
        return
    
    print(f"✅ 找到应用文件: {app_file}")
    print()
    
    # 显示网络信息
    recommended_ip = show_network_info()
    
    if recommended_ip is None:
        return
    
    # 询问是否启动
    print("="*70)
    print()
    
    try:
        choice = input("是否现在启动Streamlit？(y/n): ").strip().lower()
        
        if choice == 'y':
            print("\n" + "="*70)
            print("  🚀 启动Streamlit...")
            print("="*70)
            print()
            print("💡 提示：")
            print("  - 启动后会自动打开浏览器")
            print("  - 手机使用上面显示的Network URL访问")
            print("  - 按 Ctrl+C 可以停止服务")
            print()
            print("-"*70)
            print()
            
            # 启动Streamlit，绑定到所有网络接口
            subprocess.run([
                sys.executable, '-m', 'streamlit', 'run',
                app_file,
                '--server.address=0.0.0.0',
                '--server.port=8501'
            ])
        else:
            print("\n手动启动命令：")
            print(f"  streamlit run {app_file} --server.address=0.0.0.0")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被取消")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")

if __name__ == '__main__':
    main()


