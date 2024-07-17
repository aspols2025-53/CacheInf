import os
import sys
import dis
import re
import copy
import torch
import numpy as np
from collections.abc import Iterable
import dill
import ast
import inspect
import torch.types
import copy
import types

class DefList:
    '''To provide a common entrance for def list management
    for the conveinence of debugging.
    '''
    def __init__(self) -> None:
        self.def_list = []

    def __iter__(self):
        return self.def_list.__iter__()

    def index(self, obj):
        return self.def_list.index(obj)

    def prepend(self, obj):
        self.def_list = [obj] + self.def_list

    def append(self, obj):
        self.def_list = self.def_list + [obj]

def copy_func(f, globals=None, module=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(f.__code__, globals, name=f.__name__,
                           argdefs=f.__defaults__, closure=f.__closure__)
    # g = functools.update_wrapper(g, f)
    if module is not None:
        g.__module__ = module
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g

# dill.dumps
def try_pickle(obj, context, log):
    """_summary_

    Args:
        s (str): source code of an object
    """
    s = "pkl = dill.dumps(obj, recurse=True)"
    # s = "pkl = dill.dumps(obj)"
    m = ast.parse(s)
    co = compile(m, '<string>', 'exec')
    context["dill"] = dill
    context["obj"] = obj
    try:
        exec(co, context)
    except Exception as e:
        log(e)
    return context["pkl"]

def mainify_code(s, context=None, log=print):
    """_summary_

    Args:
        s (str): source code of an object
    """
    m = ast.parse(s)
    co = compile(m, '<string>', 'exec')
    try:
        exec(co, context)
    except Exception as e:
        log(e)

def mainify_global_data(name, val, context=None):
    """_summary_

    Args:
        name (str): global data name
        val (any): global data value
    """
    context[name] = val

def isdata(obj):
    return not inspect.isfunction(obj) and not inspect.ismodule(obj) and not inspect.ismethod(obj) and not isinstance(obj, dict.__len__.__class__) and not inspect.isclass(obj) and not inspect.isbuiltin(obj)

def getfile(obj):
    return inspect.getabsfile(inspect.unwrap(obj))

def getsource(obj):
    return inspect.getsource(inspect.unwrap(obj))

def recur_dump_obj(obj: object, log=print):
    '''Get source code for object and possibly its nested reference to other classes, global functions and data.
    First iterate its parent classes, then global functions, 
    finally class attributes and bytecode reference. We identify self-defined objects by matching the
    file location with the project location. (issue:
    https://stackoverflow.com/questions/52402783/pickle-class-definition-in-module-with-dill)
    '''
    project_dirs = []
    exclude_dirs = [os.path.dirname(os.path.abspath(__file__)),
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    extra_dirs = []
    try:
        extra_dirs.append(os.path.dirname(inspect.getabsfile(inspect.unwrap(obj.__class__))))
        extra_dirs.append(os.path.dirname(inspect.getabsfile(inspect.unwrap(obj))))
    except Exception:
        pass
    for p in sys.path + extra_dirs:
        if "dist-packages" in p or "site-packages" in p or "conda" in p or "/usr/lib" in p:
            continue
        if len(p) == 0:
            continue
        if p in project_dirs:
            continue
        if p in exclude_dirs:
            continue
        project_dirs.append(p)
    third_party_globals = {}
    log(f"Project Dirs {project_dirs}")
    def is_self_defined(obj, depth=None):
        d = 0
        try:
            while True:
                file_path = inspect.getabsfile(obj)
                for path in project_dirs:
                    if file_path.startswith(path):
                        return file_path
                if hasattr(obj, "__wrapped__"):
                    obj = obj.__wrapped__
                else:
                    break
                d += 1
                if depth and d >= depth:
                    break
        except TypeError:
            pass
        return False
    if not is_self_defined(obj) and not is_self_defined(obj.__class__):
        log("The object is not self-defined. Directly pickling...")
        pkl = dill.dumps(obj)
        log("Pickling finished.")
        return pkl
    obj = copy.deepcopy(obj)
    origin_obj = obj
    global_data = {}
    def get_obj_source(obj: object, re_rule: str) -> str:
        """Get source code using inspect.getsource(). Remove leading space if necessary.

        Args:
            obj (object): _description_
            global_name_list (list): _description_
            global_data (dict): _description_

        Returns:
            str: _description_
        """
        obj_name = obj.__name__
        # log(f"Parsing code for {obj_name}")
        try:
            src = getsource(obj).splitlines()
        except TypeError as e:
            log(str(e))
            return []
        num_start_space = re.search(r"[a-zA-Z_@]", src[0]).start()  # Remove leading space
        for i, _src in enumerate(src):
            if num_start_space > 0:
                _src = _src[num_start_space:]
            origin_src = copy.deepcopy(_src)
            matches = re.finditer(re_rule, _src)
            for match in list(matches)[::-1]:
                if "self" in _src[match.start():match.end()]:
                    continue
                _src = _src[:match.start()] + _src[match.end():]
            if len(_src) != len(origin_src):
                log(f"origin:\n\t{origin_src}\nto:\n\t{_src}")
            src[i] = _src
        return "\n".join(src)
    def get_base_class(cls, def_list: DefList):
        if hasattr(cls, "__bases__"):
            for base in cls.__bases__:
                if not is_self_defined(base):
                    continue
                if base not in def_list:
                    def_list.prepend(base)
                this_idx = def_list.index(cls)
                base_idx = def_list.index(base)
                if this_idx < base_idx:
                    def_list.prepend(base)
                def_list = get_base_class(base, def_list)
        return def_list
    def parse_func(func, def_list: DefList):
        path = is_self_defined(func)
        if not path and func != origin_obj:
            return def_list
        bytecodes = dis.Bytecode(func)
        start_line = -1
        code_line = -1
        loaded_global_module = False
        attr_list = []
        start_obj = __import__(func.__module__, fromlist=['xxx'])
        for instr in bytecodes:
            if instr.starts_line is not None:
                if start_line < 0:
                    start_line = instr.starts_line
                code_line = instr.starts_line - start_line
            if instr.opname not in ("LOAD_GLOBAL", "LOAD_ATTR"):
                loaded_global_module = False
            # Global reference or attribute of a global reference
            if (instr.opname == "LOAD_GLOBAL") or \
                (loaded_global_module and instr.opname == "LOAD_ATTR"):
                if instr.opname == "LOAD_GLOBAL":
                    loaded_global_module = True
                    attr_list = []
                if instr.argval == "super":
                    continue
                attr_list.append(instr.argval)
                try:
                    true_obj = start_obj
                    for attr in attr_list:
                        true_obj = getattr(true_obj, attr)
                except AttributeError as e:
                    continue
                if isinstance(true_obj, dict.__len__.__class__):    # Skip method wrapper
                    continue
                if (inspect.ismodule(true_obj) or (inspect.isclass(true_obj)) and not is_self_defined(true_obj)):
                    loaded_global_module = False
                    attr_list = []
                    if instr.argval not in third_party_globals:
                        third_party_globals[instr.argval] = true_obj
                        # log(f"Adding reference for {instr.argval}")
                    continue
                if inspect.isfunction(true_obj) and is_self_defined(true_obj) and true_obj not in def_list:
                    def_list.prepend(true_obj)
                    def_list = parse_func(true_obj, def_list)
                elif is_self_defined(true_obj.__class__) and true_obj.__class__ not in def_list:
                    def_list = parse_obj(true_obj, def_list)
                elif is_self_defined(true_obj) and true_obj not in def_list:
                    def_list = parse_obj(true_obj, def_list)
                elif isdata(true_obj):
                    if instr.argval in global_data and global_data[instr.argval] is not true_obj or instr.argval not in global_data:
                        # log(f"Warning: global data with the same name ({instr.argval}) may cause error.")
                        def_list = parse_obj(true_obj, def_list)
                        global_data[instr.argval] = {"func_name": func.__name__,
                                                    "data": true_obj,
                                                    "line": code_line}
        return def_list
    def parse_obj(obj: object, def_list: DefList):
        '''For class instance, parse each attribute/func
        For func, parse each global reference; ignore method wrapper
        '''
        if isinstance(obj, Iterable) and not isinstance(obj, str) and \
            hasattr(obj, "__len__") and len(obj) > 0\
            and not isinstance(obj, torch.Tensor) and not isinstance(obj, np.ndarray):
            for d in obj:
                try:
                    _d = obj[d] # dict
                    def_list = parse_obj(_d, def_list)
                except Exception:
                    pass
                def_list = parse_obj(d, def_list)

        if inspect.isfunction(obj) and is_self_defined(obj):
            if hasattr(obj,  "__closure__") and obj.__closure__ is not None:
                if hasattr(obj, "__wrapped__"):
                    wrapper = obj.__closure__[0].cell_contents
                    if is_self_defined(wrapper, 1) and wrapper not in def_list \
                        and wrapper.__name__ != wrapper.__name__:
                        def_list.prepend(wrapper)
                for clo in obj.__closure__:
                    def_list = parse_obj(clo.cell_contents, def_list)
            if is_self_defined(obj, 1):
                def_list = parse_func(obj, def_list)
        elif is_self_defined(obj.__class__):  # obj is an instance
            if obj.__class__ not in def_list:
                def_list = parse_obj(obj.__class__, def_list)
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(obj, attr_name)
                if inspect.isfunction(attr) and is_self_defined(attr) and attr not in def_list:
                    # Only parse function of an instance
                    def_list.prepend(attr)
                    def_list = parse_obj(attr, def_list)
                elif isdata(attr) and is_self_defined(attr.__class__):
                    def_list = parse_obj(attr, def_list)
        elif is_self_defined(obj):
            # obj is a self-defined clas
            if obj not in def_list:   # is a class
                def_list.prepend(obj)
                def_list = get_base_class(obj, def_list)
                for attr_name in dir(obj):
                    if attr_name.startswith("_") and attr_name != "__init__":
                        continue
                    attr = getattr(obj, attr_name)
                    if inspect.isfunction(attr) and is_self_defined(attr):
                        # class member function
                        def_list = parse_obj(attr, def_list)
                    elif isdata(attr) and is_self_defined(attr.__class__):
                        def_list = parse_obj(attr, def_list)
        return def_list
    def reset_obj_cls(obj: object, context: dict):
        """Reset obj.__class__ to the cls declared in __main__.
        We only need to reset the stored attributes.

        Args:
            obj (object): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(obj, Iterable) and not isinstance(obj, str) and len(obj) > 0\
            and not isinstance(obj, torch.Tensor) and not isinstance(obj, np.ndarray):
            for d in obj:
                try:
                    reset_obj_cls(obj[d], context)
                except Exception:
                    pass
                reset_obj_cls(d, context)

        if hasattr(obj,  "__closure__") and obj.__closure__ is not None:
            for clo in obj.__closure__:
                content = clo.cell_contents
                reset_obj_cls(content, context)
        if inspect.isfunction(obj) and is_self_defined(obj):
            if hasattr(obj, "__module__"):
                obj.__module__ = "__main__"
            if hasattr(obj, "__wrapped__"):
                reset_obj_cls(obj.__wrapped__, context)
            if hasattr(obj, "__globals__") and obj.__globals__ is not context:
                obj = copy_func(obj, context, "__main__")
            return obj
        elif is_self_defined(obj.__class__):
            # Self-defined class instance
            obj.__class__ = context[obj.__class__.__name__]
            if hasattr(obj, "__module__"):
                obj.__module__ = "__main__"
            for attr_name in dir(obj):
                attr = getattr(obj, attr_name)
                if attr_name.startswith("_"):
                    continue
                if isinstance(attr, Iterable) and len(attr) > 0:
                    reset_obj_cls(attr, context)
                if is_self_defined(attr.__class__): # instance
                    reset_obj_cls(attr, context)
                elif inspect.ismethod(attr) and is_self_defined(attr.__func__):
                    new_attr = reset_obj_cls(attr.__func__, context)
                    setattr(obj, attr_name, types.MethodType(new_attr, obj))
                elif inspect.isfunction(attr) and is_self_defined(attr):
                    new_attr = reset_obj_cls(attr, context)
                    setattr(obj, attr_name, new_attr)
                attr = getattr(obj, attr_name)
                if hasattr(attr, "__globals__") and "letterbox" in attr.__globals__:
                    assert attr.__globals__["letterbox"] == context["letterbox"]
    log("----Start parsing objects----")
    def_list = DefList()
    def_list = parse_obj(obj, def_list).def_list
    srcs = []
    name_list = []
    for _d in def_list:
        name_list.append(_d.__name__)

    for i, _n in enumerate(name_list):
        try:
            while _n is not None:
                idx = name_list[i+1:].index(_n)
                name_list[i + 1 + idx] = None
        except:
            continue
    def_list = [def_list[i] for i in range(len(name_list)) if name_list[i] is not None]
    name_list = list(filter(lambda x: x is not None, name_list))
    log("----Object parsing finished----")

    # Parse source code
    prefix = r"([a-zA-Z_0-9]*\.|\((.*?)\)\.|\[(.*?)\]\.)+"
    all_global_names = f"({'|'.join(name_list + list(global_data.keys()))})"
    suffix = r"(\s|$|[\+\-\*/%@\.\)\]])"
    re_rule = f"{prefix}(?={all_global_names}{suffix})"

    for _d in def_list:
        src = get_obj_source(_d, re_rule)
        srcs.append(src)

    # Declare self definied classes and global data in __main__
    _global_data = {}
    for key, val in global_data.items():
        _global_data[key] = val["data"]
    third_party_globals.update(_global_data)
    obj_globals = third_party_globals
    for i, src in enumerate(srcs):
        mainify_code(src, obj_globals, log=log)
    for key, val in global_data.items():
        mainify_global_data(key, val['data'], obj_globals)
    for key in name_list:
        val = obj_globals[key]
        if hasattr(val, "__module__"):
            val.__module__ = "__main__"

    reset_obj_cls(obj, obj_globals)
    log("Pickling...")
    pkl = try_pickle(obj, obj_globals, log=log)
    log("Pickling finished.")
    return pkl

