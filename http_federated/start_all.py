"""
一键启动HTTP联邦学习系统
"""
import os
import sys
import subprocess
import time

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    http_dir = os.path.dirname(os.path.abspath(__file__))
    # 修正：虚拟环境在项目根目录，不是http_federated目录
    venv_python = os.path.join(project_root, '.venv', 'Scripts', 'python.exe')

    if not os.path.exists(venv_python):
        print("[ERROR] 找不到虚拟环境!")
        print(f"路径: {venv_python}")
        input("按回车退出...")
        return

    # 解析命令行参数
    algorithm = 'fedavg'
    mu = 0.01
    privacy = 'none'
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--algorithm' and i < len(sys.argv) - 1:
            algorithm = sys.argv[i + 1]
        elif arg == '--mu' and i < len(sys.argv) - 1:
            mu = sys.argv[i + 1]
        elif arg == '--privacy' and i < len(sys.argv) - 1:
            privacy = sys.argv[i + 1]

    print("="*70)
    print("HTTP联邦学习系统 - 一键启动")
    print("="*70)
    print(f"Python路径: {venv_python}")
    print(f"算法: {algorithm.upper()}, Mu: {mu}, 隐私保护: {privacy}")
    print()

    # 启动客户端
    num_clients = 3
    clients = []
    for i in range(num_clients):
        port = 6000 + i
        print(f"[启动] 客户端 {i} (端口 {port})...")
        cmd = [venv_python, 'client.py', str(i), str(port)]
        p = subprocess.Popen(cmd, cwd=http_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
        clients.append(p)
        time.sleep(0.5)

    print()
    print("[启动] 中心服务器 (端口 5001)...")
    cmd = [venv_python, 'server.py', '--auto', 
           '--algorithm', algorithm,
           '--mu', str(mu),
           '--privacy', privacy]
    server = subprocess.Popen(cmd, cwd=http_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)

    print()
    print("="*70)
    print("[OK] 所有进程已启动!")
    print("="*70)
    print()
    print("训练将在客户端就绪后自动开始...")
    print("按 Ctrl+C 退出")
    print("="*70)

    try:
        server.wait()
        print("\n[关闭] 服务器已退出，正在检查客户端...")
        time.sleep(2)
        remaining_clients = []
        for i, p in enumerate(clients):
            if p.poll() is None:
                remaining_clients.append((i, p))
        
        if remaining_clients:
            print(f"[关闭] 强制关闭 {len(remaining_clients)} 个未退出的客户端...")
            for i, p in remaining_clients:
                try:
                    p.terminate()
                    print(f"[OK] 客户端 {i} 已关闭")
                except:
                    pass
        print("[OK] 所有进程已关闭")
    except KeyboardInterrupt:
        print("\n[关闭] 正在关闭所有进程...")
        for p in clients:
            p.terminate()
        server.terminate()
        print("[OK] 所有进程已关闭")

if __name__ == '__main__':
    main()

