# 后端和数据库配置、启动方式

## 数据库
确保已经正确安装了对应版本的MySQL和Neo4j数据库
1. 打开终端控制台
2. 运行mysql -u root -p，回车跳过输入密码（如果已经注册过密码则输入密码），进入mysql>
3. 创建数据库QMZJ：create database QMZJ;
4. 打开新的终端控制台
5. 运行neo4j.bat console（要保持终端一直打开）或者neo4j.bat start
6. 打开浏览器，访问http://localhost:7474/，进入Neo4j数据库页面，默认用户名和密码均为neo4j，登录
7. 可以自行修改Neo4j数据库密码
8. 打开源文件/backend，进入config.py文件，修改SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:password@localhost:3306/QMZJ'，password填写你的MySQL数据库的密码
9. 打开源文件/backend/db文件夹，进入config.py，修改graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo4j"))，默认的账号和密码均为neo4j

## 后端
1. 在控制台或者终端中打开源代码文件夹作为工作目录
2. 切换到backend目录：cd backend
3. 确保已经安装了Python3.9，并创建一个Python3.9的环境  
  a. 如果只安装了Python3.9，python3.9 -m venv myenv，即可创建一个名为“myenv”的python3.9虚拟环境，然后运行cd myenv/Scripts，输入activate并回车，激活虚拟环境  
  b. 如果安装了Anaconda，可以打开conda终端使用conda create -n myenv python=3.9，即可创建一个名为“myenv”的python3.9虚拟环境，然后运行conda activate myenv激活虚拟环境
4. 在虚拟环境下打开“源代码/backend”作为工作目录，运行pip install -r requirements.txt，安装所需要的Python库
5. 运行python create_table.py创建MySQL数据库表
6. 切换到db目录：cd db
7. 运行python create_graph.py创建Neo4j知识图谱
8. 回到backend目录，cd ..
9. 运行python app.py启动后端


# 使用运行说明
完成上述系统安装的步骤之后，若后续重新运行项目，则只需要完成以下步骤：
1. 启动数据库：首先确保MySQL数据库和Neo4j数据库已经启动，并且对应的用户名和密码已经在backend/config.py和backend/db/config.py中进行配置，具体如下：  
  a. 启动MySQL：在终端中输入net start mysql  
  b. 启动Neo4j：在终端中输入neo4j.bat console（需要一直保持终端打开）或者neo4j.bat start  
2. 启动后端：在控制台或者终端中打开源代码文件夹作为工作目录，切换到backend目录：cd backend，运行python app.py