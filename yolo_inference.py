from ultralytics import YOLO

model = YOLO(r"C:\Users\rupes\OneDrive\Desktop\FootballAnalysis\training\runs\detect\train12\weights\best.pt")

return_res = model.predict(r"C:\Users\rupes\OneDrive\Desktop\FootballAnalysis\data\input\08fd33_4.mp4", save = True)

print(return_res)
