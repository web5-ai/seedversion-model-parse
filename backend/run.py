'''
用于启动后台
'''

import uvicorn

if __name__ == "__main__":
    # 127.0.0.1:8000打开网页    
    # 访问127.0.0.1:8000/docs查看文档
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level='info')