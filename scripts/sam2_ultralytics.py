from ultralytics import SAM

# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
model.info()

# Run inference
# Save to a specific project and name the run
model("/home/cs24m118/datasets/videos/sam_test_pegion.mp4", save=True, project="my_results", name="pigeon_test")