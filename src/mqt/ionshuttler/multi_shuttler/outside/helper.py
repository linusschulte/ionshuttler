from outside.processing_zone import ProcessingZone

def generate_pzs(num_pzs, m, n, v, h, height = -1):
   
    if num_pzs <= 0:
        pz_definitions =  {
            "pz1": ProcessingZone(
                "pz1",
                [
                    (float((m - 1) * v), float((n - 1) * h)),
                    (float((m - 1) * v), float(0)),
                    (float((m - 1) * v - height), float((n - 1) * h / 2)),
                ],
            ),
            "pz2": ProcessingZone("pz2", [(0.0, 0.0), (0.0, float((n - 1) * h)), (float(height), float((n - 1) * h / 2))]),
            "pz3": ProcessingZone(
                "pz3", [(float((m - 1) * v), float(0)), (float(0), float(0)), (float((m - 1) * v / 2), float(height))]
            ),
            "pz4": ProcessingZone(
                "pz4",
                [
                    (float(0), float((n - 1) * h)),
                    (float((m - 1) * v), float((n - 1) * h)),
                    (float((m - 1) * v / 2), float((n - 1) * h - height)),
                ],
            ),
        }
    else:
        pz_definitions = {}
        top_segments = max(n-1, 0)
        bottom_segments = top_segments
        left_segments = max(m-1, 0)
        right_segments = left_segments
        max_segments = max(top_segments, bottom_segments, left_segments, right_segments)
        pz_idx = 1


        for idx in range(max_segments):
            if idx < top_segments:
                y_start = idx * h
                y_end = (idx + 1) * h
                pz_name = f"pz{pz_idx}"
                pz_definitions[pz_name] = ProcessingZone(
                    pz_name,
                    [
                        (0.0, float(y_start)),
                        (0.0, float(y_end)),
                        (float(height), float((y_start + y_end) / 2)),
                    ],
                )
                pz_idx += 1

            if idx < bottom_segments:
                y_start = (bottom_segments-idx) * h
                y_end = (max_segments- (idx + 1)) * h
                pz_name = f"pz{pz_idx}"
                pz_definitions[pz_name] = ProcessingZone(
                    pz_name,
                    [
                        (float((m - 1) * v), float(y_start)),
                        (float((m - 1) * v), float(y_end)),
                        (float((m - 1) * v - height), float((y_start + y_end) / 2)),
                    ],
                )
                pz_idx += 1

            if idx < left_segments:
                x_start = (left_segments - idx) * v
                x_end = (left_segments - (idx + 1)) * v
                pz_name = f"pz{pz_idx}"
                pz_definitions[pz_name] = ProcessingZone(
                    pz_name,
                    [
                        (float(x_start), 0.0),
                        (float(x_end), 0.0),
                        (float((x_start + x_end) / 2), float(height)),
                    ],
                )
                pz_idx += 1

            if idx < right_segments:
                x_start = idx * v
                x_end = (idx + 1) * v
                pz_name = f"pz{pz_idx}"
                pz_definitions[pz_name] = ProcessingZone(
                    pz_name,
                    [
                        (float(x_start), float((n - 1) * h)),
                        (float(x_end), float((n - 1) * h)),
                        (float((x_start + x_end) / 2), float((n - 1) * h - height)),
                    ],
                )
                pz_idx += 1


    return pz_definitions


def recalculate_architecture_config(meta_study_config: dict, population_density: float) -> dict:
    #if "num_ions" in meta_study_config and "grid_size" in meta_study_config:
    #    new_num_ions_list = []
    #    for grid_size in meta_study_config["grid_size"]:
    #        for mz_trap_size in meta_study_config.get("mz_trap_size", [1]):
    #            max_ions = int((2*grid_size * (grid_size-1) * mz_trap_size) * population_density)
    #            new_num_ions_list.extend([num_ions for num_ions in meta_study_config["num_ions"] if num_ions <= max_ions])
    #    meta_study_config["num_ions"] = list(set(new_num_ions_list))  # Remove duplicates
    return meta_study_config
