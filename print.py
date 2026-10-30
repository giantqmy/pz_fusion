from ultralytics import YOLO

model = YOLO(r"D:\Desktop\txt\txt2\rewrite\bestddc.pt")
print(model.model.model[1])  # P2 PZDualPIMStage 及其子模块
print(model.model.model[1].state_dict().keys())  # 该层保存的参数名