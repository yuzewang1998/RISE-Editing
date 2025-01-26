import importlib
from Editor.checkpoints_controller.base_checkpoints_controller import  BaseCheckpointsController
#对model的多继承,选择合适的model进行import
def find_checkpointcontroller_class_by_name(controller_name):
    checkpoints_controller_filename = "Editor.checkpoints_controller." + controller_name+"_checkpoints_controller" #In general: mvs_points_volumetric_model
    modellib = importlib.import_module(checkpoints_controller_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = (controller_name+"_checkpoints_controller").replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseCheckpointsController):
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of base_checkpoints_controller with class name that matches %s in lowercase."
            % (checkpoints_controller_filename, target_model_name))
        exit(0)
    return model


def create_checkpointscontroller(opt,classname,filename):
    model = find_checkpointcontroller_class_by_name(classname)
    instance = model(opt,filename)
    return instance


if __name__ == "__main__":
    pass