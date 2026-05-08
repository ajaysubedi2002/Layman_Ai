import numpy as np
class StablePlayerID:

    def __init__(self, n=None, reid_dist=140, memory=None,

                 size_ratio_min=0.45, velocity_alpha=0.7, debug=False):

        self.n              = n

        self.reid_dist      = float(reid_dist)

        self.memory         = memory

        self.size_ratio_min = float(size_ratio_min)

        self.velocity_alpha = float(velocity_alpha)

        self.debug          = debug

        self.active:   dict[int, str]  = {}

        self.profiles: dict[str, dict] = {}

        self.lost:     dict[str, dict] = {}

        self._next                     = 1

        self._frame_cursor             = -1

        self._assigned_this_frame: set[str] = set()



    def _dist(self, a, b):

        return float(np.hypot(a[0]-b[0], a[1]-b[1]))



    def _size_ratio(self, s1, s2):

        w1,h1 = max(float(s1[0]),1.0), max(float(s1[1]),1.0)

        w2,h2 = max(float(s2[0]),1.0), max(float(s2[1]),1.0)

        return min(min(w1,w2)/max(w1,w2), min(h1,h2)/max(h1,h2))



    def _predict_center(self, profile, frame):

        cx, cy = profile["c"]

        vx, vy = profile.get("v", (0.0, 0.0))

        dt = min(max(0, int(frame - profile.get("f", frame))), 30)

        return (cx + vx*dt, cy + vy*dt)



    def _touch_profile(self, label, center, size, frame, conf):

        if label not in self.profiles:

            self.profiles[label] = {

                "c": (float(center[0]), float(center[1])),

                "s": (float(size[0]),   float(size[1])),

                "f": int(frame), "v": (0.0, 0.0), "conf": float(conf),

            }

            return

        p = self.profiles[label]

        prev_cx, prev_cy = p["c"]

        dt = max(1, int(frame - p.get("f", frame)))

        raw_vx = (float(center[0]) - prev_cx) / dt

        raw_vy = (float(center[1]) - prev_cy) / dt

        prev_vx, prev_vy = p.get("v", (0.0, 0.0))

        vx = self.velocity_alpha*prev_vx + (1-self.velocity_alpha)*raw_vx

        vy = self.velocity_alpha*prev_vy + (1-self.velocity_alpha)*raw_vy

        conf_w = min(max(float(conf), 0.0), 1.0)

        pos_w  = 0.35 + 0.65*conf_w

        p["c"] = (float(prev_cx*(1-pos_w) + float(center[0])*pos_w),

                  float(prev_cy*(1-pos_w) + float(center[1])*pos_w))

        p["s"] = (float(0.8*p["s"][0] + 0.2*float(size[0])),

                  float(0.8*p["s"][1] + 0.2*float(size[1])))

        p["v"] = (float(vx), float(vy))

        p["f"] = int(frame)

        p["conf"] = float(0.8*p.get("conf", conf_w) + 0.2*conf_w)



    def _match_existing_label(self, center, size, frame):

        best_label, best_score = None, float("inf")

        for label, profile in self.profiles.items():

            if label in self._assigned_this_frame:

                continue

            pred_c     = self._predict_center(profile, frame)

            dist       = self._dist(center, pred_c)

            size_ratio = self._size_ratio(size, profile["s"])

            if size_ratio < self.size_ratio_min:

                continue

            gap  = max(0, int(frame - int(profile.get("f", frame))))

            gate = self.reid_dist * (1.0 + min(gap/45.0, 1.5))

            if dist > gate:

                continue

            score = dist + 0.35*(1.0-size_ratio)*self.reid_dist

            if score < best_score:

                best_score, best_label = score, label

        return best_label



    def get_label(self, tid, x1, y1, x2, y2, frame, conf=1.0):

        if frame != self._frame_cursor:

            self._frame_cursor = int(frame)

            self._assigned_this_frame.clear()

        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0

        size   = (max(1.0, x2-x1), max(1.0, y2-y1))

        if tid in self.active:

            label = self.active[tid]

            if label in self.profiles:

                pred_c = self._predict_center(self.profiles[label], frame)

                dist   = self._dist((cx,cy), pred_c)

                if dist <= self.reid_dist*2.2 and label not in self._assigned_this_frame:

                    self._assigned_this_frame.add(label)

                    self._touch_profile(label, (cx,cy), size, frame, conf)

                    return label

            del self.active[tid]

        label = self._match_existing_label((cx,cy), size, frame)

        if label is None:

            label = f"Player-{self._next}"

            self._next += 1

        self.active[tid] = label

        self._assigned_this_frame.add(label)

        self._touch_profile(label, (cx,cy), size, frame, conf)

        if self.debug:

            print(f"[ID] frame={frame} tid={tid} -> {label}")

        return label



    def mark_lost(self, frame, active_ids, id_centres):

        for tid in list(set(self.active) - active_ids):

            label = self.active.pop(tid)

            if label in self.profiles and tid in id_centres:

                c = id_centres[tid]

                self.profiles[label]["c"] = (float(c[0]), float(c[1]))

                self.profiles[label]["f"] = int(frame)

        active_labels = set(self.active.values())

        self.lost = {l:p for l,p in self.profiles.items() if l not in active_labels}

        if self.memory is not None:

            stale = [l for l,p in self.profiles.items()

                     if frame - int(p.get("f", frame)) > int(self.memory)]

            for l in stale:

                self.profiles.pop(l, None)

                self.lost.pop(l, None)


