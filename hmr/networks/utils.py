def get_class(class_list, name):
    class_dict = {class_.__name__.lower(): class_ for class_ in class_list}
    class_ = None
    name = name.lower()
    if name in class_dict:
        class_ = class_dict[name]
    return class_
