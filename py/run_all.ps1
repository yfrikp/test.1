#Requires -Version 5.1
<#
  ============================================================================
   题目1 算法开发套件 —— Windows 上一键跑起来（不需要 WSL / 不需要 ROS2）
  ============================================================================

  用途：
    这部分代码是【纯 Python + OpenCV】，不依赖 ROS2。
    所以即使 Ubuntu/ROS2 还没装好，你也能先把算法和阈值调通。
    等环境好了，同一套逻辑已经在 C++ 节点里实现好了（color_detector_node.cpp），
    参数可以直接搬过去。

  程序会按顺序做 4 件事：
    1) 检查 Python 与 opencv 是否可用（缺了会给出安装指引）
    2) 生成合成测试图（带精确真值）
    3) 跑离线自测，报告分类准确率 / 距离误差 / 处理速度
    4) 提示怎么打开交互式调参工具

  用法（右键 -> 使用 PowerShell 运行，或在 PowerShell 里执行）：
    powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Lenovo\dsh-workspace\vision_task1\py\run_all.ps1"
#>

$ErrorActionPreference = 'Continue'
$PyDir = Split-Path -Parent $PSCommandPath

function Sec($t) { Write-Host ''; Write-Host ('=' * 70) -ForegroundColor DarkCyan; Write-Host "  $t" -ForegroundColor Cyan; Write-Host ('=' * 70) -ForegroundColor DarkCyan }
function Ok($m)   { Write-Host "  [ OK ] $m" -ForegroundColor Green }
function Bad($m)  { Write-Host "  [FAIL] $m" -ForegroundColor Red }
function Warn2($m){ Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Info($m) { Write-Host "         $m" -ForegroundColor Gray }

Write-Host ''
Write-Host '############################################################' -ForegroundColor Magenta
Write-Host '#  题目1 算法开发套件（Windows / Python / 无需 ROS2）        #' -ForegroundColor Magenta
Write-Host '############################################################' -ForegroundColor Magenta

# ---------------------------------------------------------------------------
Sec '第 1 步 / 4   检查 Python 环境'
# ---------------------------------------------------------------------------
$python = $null
foreach ($cand in @('py', 'python', 'python3')) {
    $c = Get-Command $cand -ErrorAction SilentlyContinue
    if (-not $c) { continue }
    # 微软商店的占位符是 0 字节，要排除
    if ($c.Source -like '*WindowsApps*') {
        try { if ((Get-Item $c.Source).Length -eq 0) { Warn2 "$cand 是微软商店占位符（0 字节），跳过"; continue } } catch {}
    }
    # 真正验证能不能跑
    $v = & $cand --version 2>&1 | Select-Object -First 1
    if ($LASTEXITCODE -eq 0 -and "$v" -match 'Python\s+3') {
        $python = $cand
        Ok "找到可用 Python：$cand -> $v ($($c.Source))"
        break
    }
}

if (-not $python) {
    Bad '没有找到可用的 Python 3'
    Write-Host ''
    Write-Host '  请先安装 Python（两种方式任选）：' -ForegroundColor White
    Write-Host ''
    Write-Host '  【方式 A】官网安装包（推荐，能正常联网时用）' -ForegroundColor Yellow
    Info '打开 https://www.python.org/downloads/windows/ 下载 Windows installer (64-bit)'
    Info '安装时务必勾选 "Add python.exe to PATH"'
    Write-Host ''
    Write-Host '  【方式 B】微软商店' -ForegroundColor Yellow
    Info '开始菜单搜 "Microsoft Store" -> 搜 "Python 3.12" -> 安装'
    Write-Host ''
    Write-Host '  装完后重新运行本脚本。' -ForegroundColor White
    Read-Host '按回车退出'
    exit 1
}

# 检查 opencv
Sec '第 2 步 / 4   检查依赖（opencv-python / numpy）'
$probe = & $python -c "import cv2, numpy; print('cv2', cv2.__version__); print('numpy', numpy.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    $probe | ForEach-Object { Ok $_ }
} else {
    Warn2 '缺少 opencv-python 或 numpy，尝试自动安装...'
    Write-Host ''
    & $python -m pip install --upgrade pip
    # 用清华源加速（国内直连 pypi 很慢）
    & $python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python numpy
    $probe2 = & $python -c "import cv2, numpy; print('cv2', cv2.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Ok $probe2
    } else {
        Bad '依赖安装失败。手动执行：'
        Info "$python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python numpy"
        Read-Host '按回车退出'
        exit 1
    }
}

# ---------------------------------------------------------------------------
Sec '第 3 步 / 4   生成合成测试图 + 跑离线自测'
# ---------------------------------------------------------------------------
Push-Location $PyDir

Write-Host '  正在生成合成测试图...' -ForegroundColor Yellow
& $python make_test_images.py --out samples
if ($LASTEXITCODE -ne 0) {
    Bad '生成测试图失败'
    Pop-Location
    Read-Host '按回车退出'
    exit 1
}

Write-Host ''
Write-Host '  正在跑离线自测...' -ForegroundColor Yellow
Write-Host ''
& $python selftest.py --samples samples

Pop-Location

# ---------------------------------------------------------------------------
Sec '第 4 步 / 4   下一步：交互式调参'
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '  自测通过后，用调参工具在【真实照片】上微调阈值：' -ForegroundColor White
Write-Host ''
Write-Host ("    cd `"$PyDir`"") -ForegroundColor Cyan
Write-Host "    $python tune_hsv.py --image samples" -ForegroundColor Cyan
Write-Host ''
Write-Host '  调参窗口操作：拖滑条看掩膜 / s 保存参数 / d 切换掩膜显示 / n p 换图 / q 退出' -ForegroundColor Gray
Write-Host ''
Write-Host '  注意：' -ForegroundColor Yellow
Write-Host '    * 这个套件不依赖 ROS2，是让你【提前把算法调通】用的。' -ForegroundColor Gray
Write-Host '    * 最终提交的是 C++ 版（color_detector_node.cpp），它能满足 60Hz。' -ForegroundColor Gray
Write-Host '    * 在 tune_hsv.py 里按 s 保存的 params_tuned.yaml 可以直接搬进' -ForegroundColor Gray
Write-Host '      C++ 包的 config/params.yaml。' -ForegroundColor Gray
Write-Host ''
Read-Host '按回车退出'
