```
Input x
    ↓
[Conv 3x3, stride] → BN → ReLU
    ↓
[Conv 3x3, stride=1] → BN
    ↓
    + ←──────────────── x (skip connection)
    ↓
  ReLU
    ↓
  Output
```

### Công thức toán học:

**Main path:**
$$\text{out} = \text{BN}_2(\text{Conv}_2(\text{ReLU}(\text{BN}_1(\text{Conv}_1(x)))))$$

**Skip connection:**
- Nếu `stride=1` và `in_channels=out_channels`:
  $$\text{identity} = x$$
  
- Nếu không (cần downsample):
  $$\text{identity} = \text{BN}_{\text{down}}(\text{Conv}_{1×1}(x))$$

**Output:**
$$\text{output} = \text{ReLU}(\text{out} + \text{identity})$$

### Ý nghĩa:
- Skip connection cho phép gradient flow trực tiếp qua network
- Giúp train deep network (18, 34, 50+ layers)
- `expansion=1` cho BasicBlock (ResNet-18, 34)

---

## 2. **backbone.py - ResNet và ResNetFPN**

### 2.1 ResNet

### Kiến trúc tổng thể:
```
Input Image (3×H×W)
    ↓
Stage 1: Conv 7×7, stride=2 → BN → ReLU → MaxPool
    ↓ (64 channels, H/4 × W/4)
Stage 2: layer1 (64 channels)  → P2
    ↓ (64, H/4 × W/4)
Stage 3: layer2 (128 channels) → P3
    ↓ (128, H/8 × W/8)
Stage 4: layer3 (256 channels) → P4
    ↓ (256, H/16 × W/16)
Stage 5: layer4 (512 channels) → P5
    ↓ (512, H/32 × W/32)
```

### Công thức cho mỗi stage:

**Stage 1 (stem):**
$$x_0 = \text{MaxPool}(\text{ReLU}(\text{BN}(\text{Conv}_{7×7, s=2}(I))))$$

**Stage 2-5 (residual layers):**

Mỗi layer có nhiều blocks:
$$x_i = \text{Layer}_i(x_{i-1}) = \text{Block}_n(...\text{Block}_2(\text{Block}_1(x_{i-1})))$$

Với ResNet-18: `[2, 2, 2, 2]` blocks cho mỗi layer.

### Gradient Checkpointing:

Công thức forward với checkpointing:
$$x_i = \text{checkpoint}(\text{Layer}_i, x_{i-1})$$

**Lợi ích:** Tiết kiệm memory bằng cách không lưu intermediate activations, recompute khi backward.

---

### 2.2 ResNetFPN (Feature Pyramid Network)

### Nguyên lý:
FPN tạo ra multi-scale features để detect objects ở nhiều kích thước khác nhau.

### Kiến trúc:
```
Backbone (Bottom-up)        FPN (Top-down + Lateral)
    
C5 (512, H/32) ─────→ [1×1 Conv] ─────→ P5 (256, H/32)
     ↓                                       ↓ [Upsample 2×]
C4 (256, H/16) ─→ [1×1 Conv] ─→ (+) ─→ [3×3 Conv] → P4 (256, H/16)
     ↓                          ↑                        ↓ [Upsample 2×]
C3 (128, H/8)  ─→ [1×1 Conv] ─→ (+) ─→ [3×3 Conv] → P3 (256, H/8)
     ↓                          ↑                        ↓ [Upsample 2×]
C2 (64, H/4)   ─→ [1×1 Conv] ─→ (+) ─→ [3×3 Conv] → P2 (256, H/4)
```

---

## 3. **fpn.py - Feature Pyramid Network**

### Công thức toán học chi tiết:

**Step 1: Lateral connections (1×1 convolutions)**

$$L_i = \text{GroupNorm}(\text{Conv}_{1×1}(C_i)), \quad i \in \{2,3,4,5\}$$

- Mục đích: Reduce channels từ `[64, 128, 256, 512]` về `256` cho tất cả levels

**Step 2: Top-down pathway**

Bắt đầu từ top (P5):
$$M_5 = L_5$$

Với mỗi level thấp hơn:
$$M_i = L_i + \text{Upsample}_{2×}(M_{i+1}), \quad i \in \{4,3,2\}$$

**Upsample** sử dụng **nearest neighbor interpolation**:
$$\text{Upsample}(x)[h,w] = x[\lfloor h/2 \rfloor, \lfloor w/2 \rfloor]$$

**Step 3: Output convolutions (3×3 conv để giảm aliasing)**

$$P_i = \text{GroupNorm}(\text{Conv}_{3×3}(M_i))$$

### Tại sao dùng GroupNorm thay vì BatchNorm?

BatchNorm unstable với small batch size:
$$\text{BN}(x) = \gamma \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}} + \beta$$

GroupNorm stable hơn:
$$\text{GN}(x) = \gamma \frac{x - \mu_{\text{group}}}{\sqrt{\sigma^2_{\text{group}} + \epsilon}} + \beta$$

---

## 4. **box_utils.py - Box Encoding/Decoding**

### Nguyên lý:
Transform giữa **absolute coordinates** và **relative offsets**.

### 4.1 Box Encoding (Training target)

**Input:**
- Reference box (anchor): $(x_a, y_a, w_a, h_a)$
- Ground truth box: $(x_g, y_g, w_g, h_g)$

**Công thức:**

Tính center và size:
$$x_a^{ctr} = x_a + 0.5w_a, \quad y_a^{ctr} = y_a + 0.5h_a$$
$$x_g^{ctr} = x_g + 0.5w_g, \quad y_g^{ctr} = y_g + 0.5h_g$$

**Regression targets (normalized offsets):**

$$t_x = w_x \cdot \frac{x_g^{ctr} - x_a^{ctr}}{w_a} = 10 \cdot \frac{x_g^{ctr} - x_a^{ctr}}{w_a}$$

$$t_y = w_y \cdot \frac{y_g^{ctr} - y_a^{ctr}}{h_a} = 10 \cdot \frac{y_g^{ctr} - y_a^{ctr}}{h_a}$$

$$t_w = w_w \cdot \log\left(\frac{w_g}{w_a}\right) = 5 \cdot \log\left(\frac{w_g}{w_a}\right)$$

$$t_h = w_h \cdot \log\left(\frac{h_g}{h_a}\right) = 5 \cdot \log\left(\frac{h_g}{h_a}\right)$$

**Weights:** $w_x = w_y = 10$, $w_w = w_h = 5$ để scale các offsets.

---

### 4.2 Box Decoding (Inference prediction)

**Input:**
- Reference box: $(x_a, y_a, w_a, h_a)$
- Predicted offsets: $(t_x, t_y, t_w, t_h)$

**Công thức inverse:**

$$\hat{x}^{ctr} = \frac{t_x}{w_x} \cdot w_a + x_a^{ctr}$$

$$\hat{y}^{ctr} = \frac{t_y}{w_y} \cdot h_a + y_a^{ctr}$$

$$\hat{w} = \exp\left(\frac{t_w}{w_w}\right) \cdot w_a$$

$$\hat{h} = \exp\left(\frac{t_h}{w_h}\right) \cdot h_a$$

**Clamp $t_w$ để tránh explosion:**
$$t_w = \min(t_w, 4.135) \Rightarrow \hat{w} \leq e^{4.135/5} \cdot w_a \approx 2.3 \cdot w_a$$

**Convert về corners:**
$$\hat{x}_1 = \hat{x}^{ctr} - 0.5\hat{w}, \quad \hat{y}_1 = \hat{y}^{ctr} - 0.5\hat{h}$$
$$\hat{x}_2 = \hat{x}^{ctr} + 0.5\hat{w}, \quad \hat{y}_2 = \hat{y}^{ctr} + 0.5\hat{h}$$

---

## 5. **rpn.py - Region Proposal Network**

### 5.1 AnchorGenerator

### Nguyên lý:
Tạo ra **anchor boxes** (reference boxes) ở mỗi vị trí trên feature map.

### Công thức tạo base anchors:

Với mỗi `(size, aspect_ratio)`:

$$h = \frac{\text{size}}{\sqrt{\text{aspect\_ratio}}}$$

$$w = \text{size} \cdot \sqrt{\text{aspect\_ratio}}$$

Base anchor (centered at origin):
$$(x_1, y_1, x_2, y_2) = \left(-\frac{w}{2}, -\frac{h}{2}, \frac{w}{2}, \frac{h}{2}\right)$$

**Example:**
- `size=32`, `aspect_ratio=0.5` → $h=45.25$, $w=22.63$
- `size=32`, `aspect_ratio=1.0` → $h=32$, $w=32$
- `size=32`, `aspect_ratio=2.0` → $h=22.63$, $w=45.25$

### Công thức generate anchors trên grid:

**Shifts cho grid $(H \times W)$ với stride $s$:**

$$\text{shifts}_x = [0, s, 2s, ..., (W-1)s]$$
$$\text{shifts}_y = [0, s, 2s, ..., (H-1)s]$$

**Mesh grid:**
$$\text{Shifts} = \begin{bmatrix} 
\text{shift}_x & \text{shift}_y & \text{shift}_x & \text{shift}_y
\end{bmatrix}_{H \times W \times 4}$$

**Final anchors:**
$$\text{Anchors} = \text{Shifts}[:, \text{None}, :] + \text{BaseAnchors}[\text{None}, :, :]$$

Flatten ra: $(H \times W \times A) \times 4$ với $A$ = số anchors per location.

---

### 5.2 RPNHead

### Kiến trúc:
```
FPN Feature (256×H×W)
    ↓
[Conv 3×3, 256] + ReLU
    ↓
    ├─→ [Conv 1×1, A] → Objectness logits
    └─→ [Conv 1×1, 4A] → BBox deltas
```

**Output shapes:**
- Objectness: $(B, A, H, W)$ - binary classification
- BBox deltas: $(B, 4A, H, W)$ - regression

---

### 5.3 Anchor Assignment (Training)

### Công thức assign labels:

**Compute IoU:**
$$\text{IoU}(A_i, G_j) = \frac{|A_i \cap G_j|}{|A_i \cup G_j|}$$

**Label assignment rules:**

1. **Negative anchors** (background):
   $$\text{label}(A_i) = 0 \quad \text{if} \quad \max_j \text{IoU}(A_i, G_j) < 0.3$$

2. **Positive anchors** (foreground):
   $$\text{label}(A_i) = 1 \quad \text{if} \quad \max_j \text{IoU}(A_i, G_j) \geq 0.7$$

3. **For each GT, best matching anchor:**
   $$\text{label}(A^*_j) = 1 \quad \text{where} \quad A^*_j = \arg\max_i \text{IoU}(A_i, G_j)$$

4. **Ignore anchors** (không dùng trong training):
   $$\text{label}(A_i) = -1 \quad \text{if} \quad 0.3 \leq \max_j \text{IoU}(A_i, G_j) < 0.7$$

---

### 5.4 RPN Loss

### Sampling strategy:

Từ tất cả anchors, sample mini-batch của 256 anchors:
- 50% positive (128)
- 50% negative (128)

**Classification loss (Binary Cross Entropy):**

$$\mathcal{L}_{\text{cls}} = -\frac{1}{N_{\text{sampled}}} \sum_{i \in \text{sampled}} \left[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \right]$$

Với:
- $y_i \in \{0, 1\}$: ground truth label
- $p_i = \sigma(\text{logit}_i)$: predicted probability

**Regression loss (Smooth L1):**

$$\mathcal{L}_{\text{reg}} = \frac{1}{N_{\text{pos}}} \sum_{i \in \text{positive}} \text{SmoothL1}(t_i - t^*_i)$$

$$\text{SmoothL1}(x) = \begin{cases}
0.5x^2 / \beta & \text{if } |x| < \beta \\
|x| - 0.5\beta & \text{otherwise}
\end{cases}$$

Với $\beta = 1/9$, $t^*_i$ = encoded target từ `box_coder_encode`.

**Total RPN loss:**
$$\mathcal{L}_{\text{RPN}} = \mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{reg}}$$

(Trong code $\lambda=1$, không explicit)

---

### 5.5 Proposal Generation (Inference)

**Step 1: Decode boxes**

Áp dụng predicted deltas lên anchors:
$$\text{Proposals} = \text{BoxDecode}(\text{Anchors}, \Delta)$$

**Step 2: Clip to image**

$$x_1, x_2 = \text{clamp}(x_1, x_2, [0, W])$$
$$y_1, y_2 = \text{clamp}(y_1, y_2, [0, H])$$

**Step 3: Score và filter**

$$\text{scores} = \sigma(\text{objectness\_logits})$$

Keep top-K proposals trước NMS:
$$K_{\text{pre-NMS}} = 2000 \text{ (train)}, 1000 \text{ (test)}$$

**Step 4: Non-Maximum Suppression**

$$\text{keep} = \text{NMS}(\text{boxes}, \text{scores}, \text{threshold}=0.7)$$

NMS algorithm:
1. Sort boxes by score descending
2. Keep highest score box
3. Remove all boxes with IoU > 0.7 với box đã chọn
4. Repeat

Keep top-K sau NMS:
$$K_{\text{post-NMS}} = 1000$$

---

## 6. **roi_heads.py - RoI Heads**

### 6.1 RoIAlign (MultiScaleRoIAlign)

### Nguyên lý:
Extract features cho mỗi proposal từ FPN multi-scale features.

### Công thức:

**Level assignment:** Quyết định proposal nên lấy từ level nào

$$k = \lfloor k_0 + \log_2(\sqrt{wh}/224) \rfloor$$

Với:
- $k_0 = 4$ (base level P4)
- $w, h$: proposal width/height
- Proposal nhỏ → P2, proposal lớn → P5

**RoIAlign operation:**

1. Project proposal lên feature map level $k$ với spatial scale $s_k = 1/2^k$

2. Chia proposal thành $7 \times 7$ bins (cho Box Head) hoặc $14 \times 14$ (cho Mask Head)

3. Trong mỗi bin, sample $n$ points (sampling_ratio=2 → $2 \times 2 = 4$ points)

4. Bilinear interpolation tại mỗi sample point:

$$f(x, y) = \sum_{i,j} w_{ij} \cdot F[i, j]$$

Với weights:
$$w_{ij} = (1 - |x - i|) \cdot (1 - |y - j|)$$

5. Average pooling trong mỗi bin:

$$\text{output}[m, n] = \frac{1}{N_{\text{samples}}} \sum_{\text{samples in bin}} f(x, y)$$

**Tại sao RoIAlign tốt hơn RoIPool?**
- RoIPool quantize coordinates → misalignment
- RoIAlign dùng bilinear interpolation → không có quantization error

---

### 6.2 BoxHead

### Kiến trúc:
```
RoIAlign output (256×7×7)
    ↓ [Flatten]
(256×7×7 = 12544)
    ↓
[FC 12544 → 1024] + ReLU
    ↓
[FC 1024 → 1024] + ReLU
    ↓
    ├─→ [FC 1024 → num_classes] → Class scores
    └─→ [FC 1024 → num_classes×4] → BBox deltas
```

### Công thức:

**Feature extraction:**
$$h_1 = \text{ReLU}(\text{FC}_6(x_{\text{flatten}}))$$
$$h_2 = \text{ReLU}(\text{FC}_7(h_1))$$

**Classification:**
$$\text{scores} = \text{FC}_{\text{cls}}(h_2) \in \mathbb{R}^{C}$$

**Box regression:**
$$\Delta = \text{FC}_{\text{bbox}}(h_2) \in \mathbb{R}^{C \times 4}$$

Mỗi class có 4 deltas riêng.

---

### 6.3 Proposal Sampling (Training)

### Add GT boxes vào proposals:

$$\text{Proposals}' = \text{Proposals} \cup \text{GT\_boxes}$$

### Công thức matching:

**Compute IoU với GT:**
$$\text{IoU}_{ij} = \text{IoU}(\text{Proposal}_i, \text{GT}_j)$$

**Match mỗi proposal với GT có IoU cao nhất:**
$$j^* = \arg\max_j \text{IoU}_{ij}$$
$$\text{matched\_label}_i = \text{GT\_label}_{j^*}$$

**Label assignment:**

$$\text{label}_i = \begin{cases}
\text{matched\_label}_i & \text{if } \text{IoU}_{ij^*} \geq 0.5 \text{ (positive)} \\
0 & \text{if } \text{IoU}_{ij^*} < 0.5 \text{ (background)} \\
-1 & \text{if } 0.5 \leq \text{IoU}_{ij^*} < 0.5 \text{ (ignore, không có trong code này)}
\end{cases}$$

### Balanced sampling:

Sample mini-batch 512 proposals:
- 25% positive ($\approx$ 128)
- 75% negative ($\approx$ 384)

**Random sampling:**
$$\text{pos\_sampled} = \text{random\_choice}(\text{pos\_indices}, \min(128, |\text{pos\_indices}|))$$
$$\text{neg\_sampled} = \text{random\_choice}(\text{neg\_indices}, 512 - |\text{pos\_sampled}|)$$

---

### 6.4 Box Losses (Training)

**Classification loss (Cross Entropy):**

$$\mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(p_{ic})$$

Với:
- $y_{ic}$: one-hot label
- $p_{ic} = \text{softmax}(\text{scores}_i)_c$

**Regression loss (Smooth L1):**

Chỉ tính trên positive samples:

$$\mathcal{L}_{\text{reg}} = \frac{1}{N_{\text{pos}}} \sum_{i \in \text{positive}} \sum_{k \in \{x,y,w,h\}} \text{SmoothL1}(\Delta^k_i - \Delta^{k*}_i)$$

Với:
- $\Delta_i$: predicted deltas cho class $c_i$
- $\Delta^*_i = \text{BoxEncode}(\text{proposal}_i, \text{GT}_{j^*})$

**Class-specific regression:** Chỉ lấy deltas tương ứng với ground truth class:

$$\Delta_{\text{selected}} = \Delta[c_i] \quad \text{(not all } C \times 4 \text{ values)}$$

---

### 6.5 Postprocessing (Inference)

**Step 1: Decode boxes cho mỗi class**

$$\text{Boxes}_{ic} = \text{BoxDecode}(\text{Proposal}_i, \Delta_{ic})$$

Mỗi proposal có $C$ predicted boxes (1 cho mỗi class).

**Step 2: Clip to image boundaries**

**Step 3: Score thresholding**

Với mỗi class $c$:
$$\text{keep}_c = \{i : p_{ic} > 0.05\}$$

**Step 4: Class-wise NMS**

Áp dụng NMS riêng cho mỗi class:
$$\text{keep}_c = \text{NMS}(\text{Boxes}_c, \text{scores}_c, \text{threshold}=0.5)$$

**Step 5: Concatenate results từ tất cả classes**

---

### 6.6 MaskHead

### Kiến trúc:
```
RoIAlign output (256×14×14)
    ↓
[Conv 3×3, 256] + ReLU
    ↓
[Conv 3×3, 256] + ReLU
    ↓
[Conv 3×3, 256] + ReLU
    ↓
[Conv 3×3, 256] + ReLU
    ↓
[ConvTranspose 2×2, stride=2, 256] + ReLU  # Upsample 14×14 → 28×28
    ↓
[Conv 1×1, num_classes]  # Class-specific masks
    ↓
Output: (num_classes × 28 × 28)
```

### Công thức:

**Feature extraction:**
$$h = \text{ReLU}(\text{Conv}^{(4)}(...\text{ReLU}(\text{Conv}^{(1)}(x))))$$

**Upsampling:**
$$h' = \text{ReLU}(\text{ConvTranspose}_{2×2, s=2}(h))$$

Công thức ConvTranspose:
$$h'[i, j] = \sum_{k,l} h[k, l] \cdot W[i - 2k, j - 2l]$$

**Mask logits:**
$$M = \text{Conv}_{1×1}(h') \in \mathbb{R}^{C \times 28 \times 28}$$

---

### 6.7 Mask Target Projection

### Nguyên lý:
Project GT mask lên proposal box để tạo training target.

### Công thức:

**Input:**
- GT mask: $M_{\text{GT}} \in \{0, 1\}^{H \times W}$
- Proposal box: $(x_1, y_1, x_2, y_2)$

**Step 1: Crop mask region**

Extract region $[y_1:y_2, x_1:x_2]$ từ GT mask.

**Step 2: RoIAlign để resize về $28 \times 28$**

$$M_{\text{target}} = \text{RoIAlign}(M_{\text{GT}}, \text{proposal}, \text{output\_size}=(28, 28))$$

Sử dụng bilinear interpolation:
$$M_{\text{target}}[i, j] = \text{bilinear\_interp}(M_{\text{GT}}, x_{ij}, y_{ij})$$

**Result:** Binary mask $M_{\text{target}} \in [0, 1]^{28 \times 28}$

---

### 6.8 Mask Loss (Training)

**Binary Cross Entropy Loss:**

Chỉ tính trên positive proposals và chỉ với class tương ứng:

$$\mathcal{L}_{\text{mask}} = -\frac{1}{N_{\text{pos}}} \sum_{i \in \text{positive}} \sum_{h,w} \left[ m^*_{ihw} \log(\sigma(M_{ic_i hw})) + (1 - m^*_{ihw}) \log(1 - \sigma(M_{ic_i hw})) \right]$$

Với:
- $M_{ic_i}$: mask logits cho class $c_i$ của proposal $i$
- $m^*_{ihw}$: target mask (projected GT mask)
- $\sigma$: sigmoid function

**Class-specific loss:** Chỉ supervise mask channel tương ứng với GT class, không penalize các class khác.

---

## 7. **mask_rcnn.py - Full Model**

### Forward pass flow:
```
Input Images (B×C×H×W)
    ↓
Backbone (ResNet18 + FPN)
    ↓
Features: {P2, P3, P4, P5}
    ↓
RPN
    ↓
Proposals (N × 4) per image
    ↓
RoI Heads
    ├─→ Box Head → {boxes, scores, labels}
    └─→ Mask Head → {masks}
    ↓
Output: Detections
```