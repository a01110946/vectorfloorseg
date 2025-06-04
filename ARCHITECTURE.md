# VectorFloorSeg Architecture Document

This document outlines the architecture of the VectorFloorSeg system. It will be updated as the project progresses.

## High-Level Architecture

(To be detailed)

- Data Input: SVG Floorplans
- Preprocessing: Vector data extraction, graph construction
- Model: Two-Stream Graph Attention Network
  - Primal Stream (Line Segments)
  - Dual Stream (Regions)
  - Cross-Stream Attention
- Tasks:
  - Boundary Detection
  - Room Classification
- Output: Segmented floorplan

## Components

(To be detailed for each phase)

### Phase 1: Environment & Dependencies
- Python Virtual Environment
- PyTorch
- PyTorch Geometric (with custom modifications)
- Supporting libraries (OpenCV, NumPy, etc.)

### Phase 2: Project Structure & Pretrained Models
(Details to be added)

### Phase 3: Data Preparation
(Details to be added)

### Phase 4: Model Implementation
(Details to be added)

### Phase 5: Training & Evaluation
(Details to be added)