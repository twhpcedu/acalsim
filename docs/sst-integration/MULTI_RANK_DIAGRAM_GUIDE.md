<!--
Copyright 2023-2026 Playlab/ACAL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Multi-Rank Diagram Selection Guide

This guide helps you choose the right diagram for your IEEE 2-column paper.

## 📊 Available Diagrams

### ⭐ Recommended for IEEE 2-Column Paper

#### 1. `multi-rank-simple.puml` - **BEST FOR PAPERS**
- **What it shows**: 2 ranks, 4 GPUs (2 per rank)
- **Size**: Compact, fits in 1 column width
- **Detail level**: Shows individual GPU components, inter-rank NVLink
- **Use when**: You need to show multi-rank coordination without overwhelming detail
- **LaTeX**: `\includegraphics[width=\columnwidth]{multi-rank-simple.pdf}`

```
Structure:
  Coordinator
     ↓
  Rank 0 (GPU 0, GPU 1)  ←→  Rank 1 (GPU 2, GPU 3)
                Inter-rank NVLink
```

#### 2. `multi-rank-minimal.puml` - **MOST COMPACT**
- **What it shows**: 2 ranks as unified boxes
- **Size**: Ultra-compact, fits in narrow columns
- **Detail level**: Minimal, focuses on architecture layers
- **Use when**: Space is extremely limited, or you want to emphasize the concept
- **LaTeX**: `\includegraphics[width=\columnwidth]{multi-rank-minimal.pdf}`

```
Structure:
  Coordinator
     ↓
  [Rank 0: HostScheduler + GPUs]  ←→  [Rank 1: HostScheduler + GPUs]
                    Inter-rank NVLink
```

### 📚 For Presentations or Appendices

#### 3. `multi-rank-layers.puml`
- **What it shows**: Layered architecture (4 layers)
- **Size**: Vertical, fits 1 column
- **Use when**: Explaining component responsibilities and separation of concerns

#### 4. `multi-rank-scalable-architecture.puml` - **DETAILED**
- **What it shows**: 3 ranks, 12 GPUs with full detail
- **Size**: Wide, needs `figure*` (full page width)
- **Use when**: You have space for detailed system view (appendix, tech report)

#### 5. `multi-rank-deployment.puml`
- **What it shows**: Practical deployment with ports and commands
- **Size**: Wide, needs `figure*`
- **Use when**: Tutorial or implementation guide sections

---

## 🎨 LaTeX Integration Examples

### Single-Column Figure (Recommended)

```latex
\begin{figure}[tb]
  \centering
  \includegraphics[width=\columnwidth]{multi-rank-simple.pdf}
  \caption{Multi-rank scalable architecture with 2 SST ranks coordinating 
           4 GPUs. The external coordinator distributes workload across ranks,
           while inter-rank NVLink enables GPU-to-GPU communication across nodes.}
  \label{fig:multirank}
\end{figure}
```

### Ultra-Compact Figure (If Space is Tight)

```latex
\begin{figure}[tb]
  \centering
  \includegraphics[width=0.95\columnwidth]{multi-rank-minimal.pdf}
  \caption{Scalable multi-rank architecture showing external coordinator
           managing multiple SST ranks with inter-rank communication.}
  \label{fig:multirank_minimal}
\end{figure}
```

### Two-Column Figure (If You Need Detail)

```latex
\begin{figure*}[tb]
  \centering
  \includegraphics[width=0.85\textwidth]{multi-rank-scalable-architecture.pdf}
  \caption{Detailed multi-rank architecture with 3 SST ranks managing 12 GPUs,
           showing both intra-rank NVLink mesh and inter-rank communication paths.}
  \label{fig:multirank_detailed}
\end{figure*}
```

---

## 📏 Size Comparison

| Diagram | Width | Height | Best Fit | Detail Level |
|---------|-------|--------|----------|--------------|
| `multi-rank-minimal` | Narrow | Short | 1-column | ⭐ (Minimal) |
| `multi-rank-simple` | Medium | Medium | 1-column | ⭐⭐⭐ (Balanced) |
| `multi-rank-layers` | Medium | Tall | 1-column | ⭐⭐ (Conceptual) |
| `multi-rank-scalable-architecture` | Wide | Medium | 2-column | ⭐⭐⭐⭐⭐ (Full) |
| `multi-rank-deployment` | Wide | Tall | 2-column | ⭐⭐⭐⭐ (Practical) |

---

## 🎯 Decision Tree

```
Q: Do you have space for a full-width figure (figure*)?
├─ YES → Use multi-rank-scalable-architecture.pdf (12 GPUs, full detail)
└─ NO (single column only)
   │
   Q: Do you already have a detailed single-rank diagram?
   ├─ YES → Use multi-rank-simple.pdf (2 ranks, 4 GPUs) ⭐ RECOMMENDED
   └─ NO → Consider using both:
           1. Single-rank detailed diagram from hybrid-multi-gpu-architecture.pdf
           2. Multi-rank minimal diagram for scalability discussion
```

---

## 💡 Recommended Combination for Paper

If you have limited space, use this 2-figure approach:

1. **Figure N**: Single-rank architecture (`hybrid-multi-gpu-architecture.pdf`)
   - Caption: "Hybrid ACALSim-SST architecture for multi-GPU simulation with dual-port design."
   - Section: Architecture Overview

2. **Figure N+1**: Multi-rank scaling (`multi-rank-simple.pdf`)
   - Caption: "Multi-rank deployment showing scalability to distributed systems."
   - Section: Scalability / Multi-Node Deployment

This gives readers:
- Detail on how a single rank works (Figure N)
- Understanding of how to scale to multiple nodes (Figure N+1)

---

## 🔧 Generating Diagrams

```bash
cd /path/to/acalsim/docs/sst-integration

# Generate all diagrams (PNG + PDF)
./generate-multi-rank-diagrams.sh

# Or generate specific diagram manually
plantuml -tpdf multi-rank-simple.puml
```

---

## 📝 Caption Templates

### For `multi-rank-simple.pdf`:

> **Short**: "Multi-rank architecture scaling across 2 SST ranks with inter-rank NVLink communication."

> **Medium**: "Multi-rank scalable architecture with 2 SST ranks coordinating 4 GPUs. The external coordinator distributes workload across ranks, while inter-rank NVLink enables cross-node GPU communication."

> **Detailed**: "Multi-rank architecture for distributed GPU simulation. An external coordinator process distributes workload across multiple SST ranks (MPI processes), each managing local GPUs. Inter-rank NVLink communication (solid red) models cross-node interconnects (~1-10 μs latency), while intra-rank links (dashed red) model NVSwitch (~100 ps)."

### For `multi-rank-minimal.pdf`:

> "Scalable multi-rank architecture showing external coordinator managing multiple SST ranks with inter-rank communication. Each rank handles local GPU simulation and scheduling."

---

## Summary

✅ **For most IEEE 2-column papers**: Use `multi-rank-simple.pdf`  
✅ **For extremely tight space**: Use `multi-rank-minimal.pdf`  
✅ **For appendix or tech report**: Use `multi-rank-scalable-architecture.pdf`

