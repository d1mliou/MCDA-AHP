"""
Ανάλυση Καταλληλότητας Χωροθέτησης Αιολικού Πάρκου — Νήσος Κύθνος
Μεθοδολογία GIS-MCDA με στάθμιση AHP και ανάλυση ευαισθησίας OAT
Εκτέλεση μέσα από την κονσόλα Python του QGIS
Συγγραφέας: Λιούμης Δημήτριος
"""

import processing
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

# ── Ονόματα layers στο QGIS project (αλλάξτε αν χρειάζεται) ──

project_layers = {
    'dem':         'DEM',
    'wind':        'Ailiko_Dynamiko_IDW',
    'viewshed':    'Viewshed',
    'settlements': 'Oikismoi_Oloi',
    'hydro':       'Ydrografiko_Diktyo',
    'roads':       'Roads_Kythnos',
    'grid':        'Ypostathmos',
    'natura':      'Natura_2021',
    'lithology':   'lithology',
}

# ── Παράμετροι περιορισμών ──

constraint_params = {
    'settlements_buffer': 1000,
    'hydro_buffer':       200,
    'slope_threshold':    25,
}

# ── Πίνακες επαναταξινόμησης (κλίμακα 1-5) ──
# Μορφή: [min1, max1, τιμή1, min2, max2, τιμή2, ...]

table_wind  = [0, 4, 1, 4, 5, 2, 5, 6, 3, 6, 7, 4, 7, 50, 5]
table_slope = [-1, 10, 5, 10, 20, 4, 20, 25, 3, 25, 30, 2, 30, 90, 1]
table_roads = [-1, 250, 5, 250, 500, 4, 500, 1000, 3, 1000, 1500, 2, 1500, 50000, 1]
table_grid  = [-1, 500, 5, 500, 1000, 4, 1000, 2000, 3, 2000, 3000, 2, 3000, 50000, 1]
table_view  = [-1, 0, 5, 0, 2, 3, 2, 3, 1]

# ── Βάρη AHP (CR = 0,03) ──

base_weights = {
    'wind':  0.41,
    'slope': 0.30,
    'roads': 0.07,
    'grid':  0.09,
    'view':  0.13
}

# ═══════════════════════════════════════════════════
# Βοηθητικές συναρτήσεις
# ═══════════════════════════════════════════════════

def get_project_layer(layer_name):
    layers = QgsProject.instance().mapLayersByName(layer_name)
    if layers:
        return layers[0]
    print(f"ΣΦΑΛΜΑ: Το layer '{layer_name}' δεν βρέθηκε στο project.")
    return None


def load_layer_safe(output_result, layer_name, is_vector=False):
    layer = None
    if isinstance(output_result, str):
        if is_vector:
            layer = QgsVectorLayer(output_result, layer_name, "ogr")
        else:
            layer = QgsRasterLayer(output_result, layer_name)
    else:
        layer = output_result
        layer.setName(layer_name)

    if layer and layer.isValid():
        QgsProject.instance().addMapLayer(layer)
        return layer
    print(f"ΣΦΑΛΜΑ: Αποτυχία φόρτωσης '{layer_name}'.")
    return None


def do_reclass(input_layer, table, out_name):
    print(f"  Επαναταξινόμηση: {out_name}...")
    try:
        res = processing.run("native:reclassifybytable", {
            'INPUT_RASTER': input_layer, 'RASTER_BAND': 1, 'TABLE': table,
            'NO_DATA': -9999, 'RANGE_BOUNDARIES': 0, 'NODATA_FOR_MISSING': False,
            'DATA_TYPE': 5, 'OUTPUT': 'TEMPORARY_OUTPUT'
        })['OUTPUT']
        return load_layer_safe(res, out_name)
    except Exception as e:
        print(f"  Σφάλμα στο {out_name}: {e}")
        return None


def align_raster(input_raster, reference_layer, out_name):
    """Ευθυγράμμιση raster με βάση το CRS, extent και ανάλυση ενός reference layer."""
    try:
        res = processing.run("gdal:warpreproject", {
            'INPUT': input_raster,
            'SOURCE_CRS': None,
            'TARGET_CRS': reference_layer.crs().authid(),
            'RESAMPLING': 0,
            'NODATA': -9999,
            'TARGET_RESOLUTION': reference_layer.rasterUnitsPerPixelX(),
            'TARGET_EXTENT': reference_layer.extent(),
            'OUTPUT': 'TEMPORARY_OUTPUT'
        })['OUTPUT']
        layer = QgsRasterLayer(res, out_name)
        if layer.isValid():
            QgsProject.instance().addMapLayer(layer)
            return layer
        print(f"  ΣΦΑΛΜΑ: Ευθυγράμμιση '{out_name}' απέτυχε.")
        return None
    except Exception as e:
        print(f"  ΣΦΑΛΜΑ ευθυγράμμισης '{out_name}': {e}")
        return None


# ═══════════════════════════════════════════════════
# Φάση 1 — Χαρτογράφηση περιορισμών
# ═══════════════════════════════════════════════════

def build_constraints(dem_layer, settlements_layer, hydro_layer, natura_layer,
                      lithology_layer, params):
    """
    Παράγει το τελικό δυαδικό raster περιορισμών (0=ακατάλληλο, 1=κατάλληλο).
    Buffer → Merge → Dissolve → Rasterize → πολλαπλασιασμός με δυαδικό κλίσης.
    """
    print("\n" + "=" * 50)
    print("ΔΗΜΙΟΥΡΓΙΑ ΠΕΡΙΟΡΙΣΜΩΝ")
    print("=" * 50)

    extent = dem_layer.extent()
    extent_str = (f"{extent.xMinimum()},{extent.xMaximum()},"
                  f"{extent.yMinimum()},{extent.yMaximum()} "
                  f"[{dem_layer.crs().authid()}]")
    pixel_size = dem_layer.rasterUnitsPerPixelX()

    # Buffer οικισμών
    print(f"  1/6 Buffer οικισμών ({params['settlements_buffer']} m)...")
    settle_buffer = processing.run("native:buffer", {
        'INPUT': settlements_layer, 'DISTANCE': params['settlements_buffer'],
        'SEGMENTS': 40, 'DISSOLVE': True, 'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Buffer υδρογραφικού
    print(f"  2/6 Buffer υδρογραφικού ({params['hydro_buffer']} m)...")
    hydro_buffer = processing.run("native:buffer", {
        'INPUT': hydro_layer, 'DISTANCE': params['hydro_buffer'],
        'SEGMENTS': 40, 'DISSOLVE': True, 'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Συγχώνευση όλων των διανυσματικών περιορισμών
    print("  3/6 Συγχώνευση (Merge)...")
    merged = processing.run("native:mergevectorlayers", {
        'LAYERS': [settle_buffer, hydro_buffer, natura_layer, lithology_layer],
        'CRS': dem_layer.crs(), 'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Dissolve επικαλυπτόμενων πολυγώνων
    print("  4/6 Dissolve...")
    dissolved = processing.run("native:dissolve", {
        'INPUT': merged, 'FIELD': [], 'SEPARATE_DISJOINT': False,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Ψηφιδοποίηση: εντός πολυγώνων = 0, εκτός = 1
    print("  5/6 Ψηφιδοποίηση (burn=0, init=1)...")
    vector_constr_raster = processing.run("gdal:rasterize", {
        'INPUT': dissolved, 'BURN': 0, 'UNITS': 1,
        'WIDTH': pixel_size, 'HEIGHT': pixel_size,
        'EXTENT': extent_str, 'NODATA': -9999,
        'DATA_TYPE': 0, 'INIT': 1, 'INVERT': False,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    vector_constr_layer = load_layer_safe(vector_constr_raster, "Constraint_Vectors")

    # Κλίση → δυαδικός χάρτης (κλίση <= κατώφλι → 1)
    print(f"  6/6 Περιορισμός κλίσης (<= {params['slope_threshold']}°)...")
    slope_res = processing.run("native:slope", {
        'INPUT': dem_layer, 'Z_FACTOR': 1, 'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    slope_layer = load_layer_safe(slope_res, "SLOPE")

    if not slope_layer or not vector_constr_layer:
        print("  ΣΦΑΛΜΑ: Αδυναμία δημιουργίας κλίσης ή διανυσματικών περιορισμών.")
        return None

    slope_binary_res = processing.run("qgis:rastercalculator", {
        'EXPRESSION': f"(\"{slope_layer.name()}@1\" <= {params['slope_threshold']}) * 1",
        'LAYERS': [slope_layer], 'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    slope_binary = load_layer_safe(slope_binary_res, "Constraint_Slope")

    if not slope_binary:
        return None

    # Τελική μάσκα = διανυσματικοί περιορισμοί × κλίση (AND)
    print("  Συνδυασμός περιορισμών (AND)...")
    combined_res = processing.run("qgis:rastercalculator", {
        'EXPRESSION': f"\"{vector_constr_layer.name()}@1\" * \"{slope_binary.name()}@1\"",
        'LAYERS': [vector_constr_layer, slope_binary],
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    combined = load_layer_safe(combined_res, "Constraints_Combined")
    if combined:
        print("  Οι περιορισμοί δημιουργήθηκαν επιτυχώς.")
    return combined


# ═══════════════════════════════════════════════════
# Φάση 2 — Κριτήρια (ψηφιδωτά απόστασης)
# ═══════════════════════════════════════════════════

def build_proximity_raster(vector_layer, dem_layer, out_name):
    """Ψηφιδοποίηση και υπολογισμός ευκλείδειας απόστασης."""
    print(f"  Απόσταση: {out_name}...")
    extent = dem_layer.extent()
    extent_str = (f"{extent.xMinimum()},{extent.xMaximum()},"
                  f"{extent.yMinimum()},{extent.yMaximum()} "
                  f"[{dem_layer.crs().authid()}]")
    pixel_size = dem_layer.rasterUnitsPerPixelX()

    rasterized = processing.run("gdal:rasterize", {
        'INPUT': vector_layer, 'BURN': 1, 'UNITS': 1,
        'WIDTH': pixel_size, 'HEIGHT': pixel_size,
        'EXTENT': extent_str, 'NODATA': 0, 'DATA_TYPE': 5,
        'INIT': 0, 'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    prox_res = processing.run("gdal:proximity", {
        'INPUT': rasterized, 'BAND': 1, 'VALUES': '1',
        'UNITS': 0, 'MAX_DISTANCE': 0, 'NODATA': -9999,
        'DATA_TYPE': 5, 'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    return load_layer_safe(prox_res, out_name)


# ═══════════════════════════════════════════════════
# Φάση 3 & 4 — Σταθμισμένη υπέρθεση
# ═══════════════════════════════════════════════════

def compute_weighted_overlay(reclass_layers, weights, scenario_name):
    expr = (
        f"({weights['wind']:.4f} * \"{reclass_layers['wind'].name()}@1\") + "
        f"({weights['slope']:.4f} * \"{reclass_layers['slope'].name()}@1\") + "
        f"({weights['roads']:.4f} * \"{reclass_layers['roads'].name()}@1\") + "
        f"({weights['grid']:.4f} * \"{reclass_layers['grid'].name()}@1\") + "
        f"({weights['view']:.4f} * \"{reclass_layers['view'].name()}@1\")"
    )
    try:
        res = processing.run("qgis:rastercalculator", {
            'EXPRESSION': expr,
            'LAYERS': list(reclass_layers.values()),
            'OUTPUT': 'TEMPORARY_OUTPUT'
        })['OUTPUT']
        return load_layer_safe(res, f"WOL_{scenario_name}")
    except Exception as e:
        print(f"  Σφάλμα στη σταθμισμένη υπέρθεση ({scenario_name}): {e}")
        return None


# ═══════════════════════════════════════════════════
# Φάση 5 — Εφαρμογή περιορισμών & εξαγωγή ζωνών
# ═══════════════════════════════════════════════════

def apply_constraints_and_extract_sites(weighted_layer, constr_layer, scenario_name):
    """
    Εφαρμογή μάσκας, κατώφλι >= 3.5, πολυγωνοποίηση,
    dissolve γειτονικών, φιλτράρισμα εμβαδού >= 0,5 km².
    Επιστρέφει (χάρτης_καταλληλότητας, ζώνες_χωροθέτησης).
    """

    # Πολλαπλασιασμός με μάσκα + αποκοπή τιμών > 5 (στρογγυλοποίηση)
    final_expr = (
        f"(\"{weighted_layer.name()}@1\" * \"{constr_layer.name()}@1\") * "
        f"(\"{weighted_layer.name()}@1\" <= 5.01)"
    )
    try:
        res = processing.run("qgis:rastercalculator", {
            'EXPRESSION': final_expr,
            'LAYERS': [weighted_layer, constr_layer],
            'OUTPUT': 'TEMPORARY_OUTPUT'
        })['OUTPUT']
        final_map = load_layer_safe(res, f"Suitability_{scenario_name}")
    except Exception as e:
        print(f"  Σφάλμα εφαρμογής περιορισμών ({scenario_name}): {e}")
        return None, None

    if not final_map:
        return None, None

    # Δυαδικό raster: ψηφίδες >= 3.5 → 1
    print(f"  Εντοπισμός ζωνών ({scenario_name})...")
    res_binary = processing.run("qgis:rastercalculator", {
        'EXPRESSION': f"\"{final_map.name()}@1\" >= 3.5",
        'LAYERS': [final_map], 'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    binary_layer = QgsRasterLayer(res_binary, f"Binary_{scenario_name}")
    if not binary_layer.isValid():
        print(f"  ΣΦΑΛΜΑ: Μη έγκυρο δυαδικό raster ({scenario_name}).")
        return final_map, None
    QgsProject.instance().addMapLayer(binary_layer)

    # Πολυγωνοποίηση
    res_poly = processing.run("native:pixelstopolygons", {
        'INPUT_RASTER': binary_layer, 'BAND': 1, 'FIELD_NAME': 'Score',
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Μόνο κατάλληλες ψηφίδες (Score = 1)
    res_suitable = processing.run("native:extractbyexpression", {
        'INPUT': res_poly, 'EXPRESSION': '"Score" = 1',
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Dissolve γειτονικών πολυγώνων (ξεχωριστά τα μη γειτονικά)
    res_dissolved = processing.run("native:dissolve", {
        'INPUT': res_suitable, 'FIELD': [], 'SEPARATE_DISJOINT': True,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    # Φιλτράρισμα εμβαδού >= 500.000 m² (0,5 km²)
    res_sites = processing.run("native:extractbyexpression", {
        'INPUT': res_dissolved, 'EXPRESSION': '$area >= 500000',
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']

    sites_layer = load_layer_safe(res_sites, f"Sites_{scenario_name}", is_vector=True)
    return final_map, sites_layer


# ═══════════════════════════════════════════════════
# Ανάλυση ευαισθησίας — OAT ±5%
# ═══════════════════════════════════════════════════

def generate_sensitivity_scenarios(base_w, perturbation=0.05):
    """
    Μεταβολή κάθε βάρους κατά ±5%, αναλογική αναπροσαρμογή
    υπολοίπων ώστε το άθροισμα να παραμείνει 1.
    """
    scenarios = []
    criteria = list(base_w.keys())

    for criterion in criteria:
        for direction, label in [(+perturbation, "+5%"), (-perturbation, "-5%")]:
            new_w = {}
            w_new = max(0.01, min(0.99, base_w[criterion] + direction))
            new_w[criterion] = w_new

            remaining_orig = sum(base_w[c] for c in criteria if c != criterion)
            remaining_new = 1.0 - w_new

            for c in criteria:
                if c != criterion:
                    new_w[c] = base_w[c] * (remaining_new / remaining_orig)

            scenarios.append((f"{criterion}_{label}", new_w))

    return scenarios


def run_sensitivity_analysis(baseline_map, reclass_layers, constr_layer,
                             base_w, perturbation=0.05):
    """
    Εκτέλεση σεναρίων OAT και υπολογισμός MACR σε σχέση με το baseline.
    MACR = MO(|S_σεναρίου − S_βάσης| / S_βάσης) × 100%
    Όριο σταθερότητας: MACR <= 5%.
    """
    print("\n" + "=" * 70)
    print("ΑΝΑΛΥΣΗ ΕΥΑΙΣΘΗΣΙΑΣ (OAT ±5%)")
    print("=" * 70)

    scenarios = generate_sensitivity_scenarios(base_w, perturbation)
    results = []

    for name, weights in scenarios:
        print(f"\n--- Σενάριο: {name} ---")
        print(f"    Βάρη: { {k: round(v, 4) for k, v in weights.items()} }")
        print(f"    Άθροισμα: {sum(weights.values()):.4f}")

        wol = compute_weighted_overlay(reclass_layers, weights, name)
        if not wol:
            continue

        suit, sites = apply_constraints_and_extract_sites(wol, constr_layer, name)
        if not suit:
            continue

        # MACR: ποσοστιαία μεταβολή ανά ψηφίδα (μόνο όπου baseline > 0.5)
        macr_expr = (
            f"( (\"{baseline_map.name()}@1\" > 0.5) * "
            f"  ( abs(\"{suit.name()}@1\" - \"{baseline_map.name()}@1\") / "
            f"    \"{baseline_map.name()}@1\" ) * 100 ) + "
            f"( (\"{baseline_map.name()}@1\" <= 0.5) * 0 )"
        )

        try:
            res_diff = processing.run("qgis:rastercalculator", {
                'EXPRESSION': macr_expr,
                'LAYERS': [suit, baseline_map],
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']

            diff_layer = QgsRasterLayer(res_diff, f"MACR_{name}")
            if not diff_layer.isValid():
                print(f"    ΣΦΑΛΜΑ: Μη έγκυρο MACR layer.")
                continue

            stats = processing.run("native:rasterlayerstatistics", {
                'INPUT': diff_layer, 'BAND': 1
            })

            mean_change = stats.get('MEAN')
            max_change = stats.get('MAX')

            if mean_change is not None:
                stable = mean_change <= 5.0
                label = "ΣΤΑΘΕΡΟ" if stable else "ΜΗ ΣΤΑΘΕΡΟ"
                print(f"    MACR (μέσος): {mean_change:.2f}%")
                print(f"    Μέγιστη μεταβολή: {max_change:.2f}%")
                print(f"    Αξιολόγηση: {label}")

                results.append({
                    'scenario': name, 'weights': weights,
                    'MACR_mean': mean_change, 'MACR_max': max_change,
                    'stable': stable
                })
        except Exception as e:
            print(f"    ΣΦΑΛΜΑ υπολογισμού MACR: {e}")

    # Πίνακας αποτελεσμάτων
    print("\n" + "=" * 70)
    print("ΣΥΝΟΨΗ ΑΝΑΛΥΣΗΣ ΕΥΑΙΣΘΗΣΙΑΣ")
    print("=" * 70)
    print(f"{'Σενάριο':<25} {'MACR (%)':>10} {'Max (%)':>10} {'Αξιολόγηση':>15}")
    print("-" * 65)

    all_stable = True
    for r in results:
        tag = "ΣΤΑΘΕΡΟ" if r['stable'] else "ΜΗ ΣΤΑΘΕΡΟ"
        print(f"{r['scenario']:<25} {r['MACR_mean']:>10.2f} {r['MACR_max']:>10.2f} {tag:>15}")
        if not r['stable']:
            all_stable = False

    print("-" * 65)
    if all_stable:
        print("ΣΥΜΠΕΡΑΣΜΑ: Το μοντέλο είναι ΣΤΑΘΕΡΟ σε μεταβολές βαρών ±5%.")
    else:
        unstable = [r['scenario'] for r in results if not r['stable']]
        print(f"ΣΥΜΠΕΡΑΣΜΑ: ΑΣΤΑΣΘΕΙΑ στα σενάρια: {', '.join(unstable)}")
    print("=" * 70)

    return results


# ═══════════════════════════════════════════════════
# Κύρια εκτέλεση
# ═══════════════════════════════════════════════════

print("=" * 70)
print("ΑΝΑΛΥΣΗ ΚΑΤΑΛΛΗΛΟΤΗΤΑΣ ΧΩΡΟΘΕΤΗΣΗΣ ΑΙΟΛΙΚΟΥ ΠΑΡΚΟΥ — ΚΥΘΝΟΣ")
print("=" * 70)

# Φόρτωση layers
print("\nΦόρτωση layers...")
dem         = get_project_layer(project_layers['dem'])
wind        = get_project_layer(project_layers['wind'])
viewshed    = get_project_layer(project_layers['viewshed'])
settlements = get_project_layer(project_layers['settlements'])
hydro       = get_project_layer(project_layers['hydro'])
roads       = get_project_layer(project_layers['roads'])
grid_point  = get_project_layer(project_layers['grid'])
natura      = get_project_layer(project_layers['natura'])
lithology   = get_project_layer(project_layers['lithology'])

all_inputs = [dem, wind, viewshed, settlements, hydro, roads, grid_point, natura, lithology]
if not all(all_inputs):
    missing = [n for n, l in zip(project_layers.values(), all_inputs) if not l]
    print(f"\nΣΦΑΛΜΑ: Λείπουν τα layers: {', '.join(missing)}")
    print("Βεβαιωθείτε ότι είναι φορτωμένα με τα σωστά ονόματα.")
else:
    print("Όλα τα layers βρέθηκαν.\n")

    # Φάση 1: Περιορισμοί
    constraints_raster = build_constraints(
        dem, settlements, hydro, natura, lithology, constraint_params
    )

    if not constraints_raster:
        print("ΣΦΑΛΜΑ: Αποτυχία δημιουργίας περιορισμών.")
    else:
        # Φάση 2: Κριτήρια
        print("\nΔΗΜΙΟΥΡΓΙΑ ΚΡΙΤΗΡΙΩΝ ΚΑΤΑΛΛΗΛΟΤΗΤΑΣ")
        print("-" * 50)
        slope_raster = get_project_layer("SLOPE")
        if not slope_raster:
            slope_res = processing.run("native:slope", {
                'INPUT': dem, 'Z_FACTOR': 1, 'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
            slope_raster = load_layer_safe(slope_res, "SLOPE")

        roads_proximity = build_proximity_raster(roads, dem, "Roads_Proximity")
        grid_proximity  = build_proximity_raster(grid_point, dem, "Ypostathmos_Proximity")

        if slope_raster and roads_proximity and grid_proximity:

            # Φάση 3: Επαναταξινόμηση
            print("\nΕΠΑΝΑΤΑΞΙΝΟΜΗΣΗ ΚΡΙΤΗΡΙΩΝ")
            print("-" * 50)
            r_wind  = do_reclass(wind, table_wind, "Aioliko_Dynamiko_Reclassified")
            r_slope = do_reclass(slope_raster, table_slope, "SLOPE_Reclass")
            r_roads = do_reclass(roads_proximity, table_roads, "Roads_Proximity_Reclass")
            r_grid  = do_reclass(grid_proximity, table_grid, "Ypostathmos_Proximity_Reclass")
            r_view  = do_reclass(viewshed, table_view, "Viewshed_Reclassified")

            if r_wind and r_slope and r_roads and r_grid and r_view:

                reclass_layers = {
                    'wind': r_wind, 'slope': r_slope, 'roads': r_roads,
                    'grid': r_grid, 'view': r_view
                }

                # Φάση 4: Σταθμισμένη υπέρθεση (baseline)
                print("\nΣΤΑΘΜΙΣΜΕΝΗ ΥΠΕΡΘΕΣΗ (BASELINE)")
                print("-" * 50)
                weighted = compute_weighted_overlay(reclass_layers, base_weights, "Baseline")

                if weighted:
                    # Φάση 5: Εφαρμογή περιορισμών & εξαγωγή ζωνών
                    print("\nΕΦΑΡΜΟΓΗ ΠΕΡΙΟΡΙΣΜΩΝ & ΕΞΑΓΩΓΗ ΖΩΝΩΝ")
                    print("-" * 50)
                    constr_aligned = align_raster(constraints_raster, weighted, "Constraints_Aligned")

                    if constr_aligned:
                        baseline_suit, baseline_sites = apply_constraints_and_extract_sites(
                            weighted, constr_aligned, "Baseline"
                        )

                        if baseline_suit:
                            print("\n" + "=" * 70)
                            print("Η ΒΑΣΙΚΗ ΑΝΑΛΥΣΗ ΟΛΟΚΛΗΡΩΘΗΚΕ ΕΠΙΤΥΧΩΣ")
                            print("=" * 70)

                            # Ανάλυση ευαισθησίας
                            sensitivity_results = run_sensitivity_analysis(
                                baseline_suit, reclass_layers,
                                constr_aligned, base_weights, perturbation=0.05
                            )
                        else:
                            print("ΣΦΑΛΜΑ: Αποτυχία βασικής ανάλυσης καταλληλότητας.")
                    else:
                        print("ΣΦΑΛΜΑ: Αποτυχία ευθυγράμμισης περιορισμών.")
                else:
                    print("ΣΦΑΛΜΑ: Αποτυχία σταθμισμένης υπέρθεσης.")
            else:
                print("ΣΦΑΛΜΑ: Αποτυχία επαναταξινόμησης.")
        else:
            print("ΣΦΑΛΜΑ: Αποτυχία δημιουργίας ψηφιδωτών απόστασης.")
