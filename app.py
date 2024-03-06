import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from way import graph, find_shortest_path_by_name, get_path_edges

app = FastAPI()

# Mount static files. FastAPI will automatically serve the files in the "static" directory under "/static"
app.mount("/static", StaticFiles(directory="static"), name="static")

# Endpoint to serve index.html
@app.get("/")
async def get_index():
    return FileResponse('static/index.html')

# Endpoint to find and return a path between two edges in JSON format
@app.get("/path")
async def get_path(start_edge: str, end_edge: str):
    # Here you would invoke your graph building and pathfinding functions
    # For the sake of this example, let's assume 'graph' is already constructed and available

    # Call your function to find the shortest path
    path, length = find_shortest_path_by_name(graph, start_edge, end_edge)

    if path:
        # Convert node IDs to edge names for the path
        edges = get_path_edges(graph, path)
    else:
        # Raise an HTTP exception if no path is found
        raise HTTPException(status_code=404, detail="Path not found")

    return edges
