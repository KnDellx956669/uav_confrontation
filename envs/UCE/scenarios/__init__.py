import importlib.util
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)

    # 获取模块的名称和路径
    module_name = osp.splitext(osp.basename(name))[0]

    # 使用 importlib.util.spec_from_file_location 加载模块
    spec = importlib.util.spec_from_file_location(module_name, pathname)

    # 创建一个新的模块对象
    module = importlib.util.module_from_spec(spec)

    # 执行模块的代码
    spec.loader.exec_module(module)

    return module
