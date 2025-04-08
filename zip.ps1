# 项目路径
$projectPath = "E:\Proj\PyProj\seedversion-model-parse\"

# 打包路径及文件名
$zipFileName = "seedversion-model-parse.tar"
$zipFilePath = "$projectPath$zipFileName"

# 目标服务器信息
$serverUser = "root" # 服务器用户名
$serverIp = "47.100.53.207" # 服务器IP地址
$serverPath = "/home/www/" # 服务器目标路径
# 检查路径是否存在

# 打包 太慢了就手动打包了
# Compress-Archive -Path $projectPath\* -DestinationPath $zipFilePath -Force

# 上传到服务器
scp $zipFilePath "$($serverUser)@$($serverIp):$($serverPath)"
scp "unzip.sh" "$($serverUser)@$($serverIp):$($serverPath)"
# 执行解压命令
ssh "$($serverUser)@$($serverIp)" "bash $($serverPath)/unzip.sh"