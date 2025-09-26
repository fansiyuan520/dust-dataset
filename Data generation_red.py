import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import random
import math
import os
import shutil
from scipy.ndimage import gaussian_filter
import colorsys


class EnhancedVisualDustSimulator:
    """
    Enhanced Visual Effect-Optimized Dust Simulator
    """

    def __init__(self, dpi=300):
        self.dpi = dpi
        self.dust_bulk_density = 1.5e6
        self.extinction_coef = 1.2
        self.visual_enhancement = True

        self.clean_image = None
        self.width = 0
        self.height = 0

    def load_image(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.clean_image = Image.open(image_path)
        self.width, self.height = self.clean_image.size
        return self.clean_image

    def enhanced_concentration_to_intensity(self, concentration_g_m2):

        if concentration_g_m2 <= 0:
            return 0.0

        thickness_m = concentration_g_m2 / self.dust_bulk_density
        thickness_um = thickness_m * 1e6
        optical_depth = self.extinction_coef * thickness_um / 1000


        optical_depth = min(optical_depth, 709)
        opacity = 1 - math.exp(-optical_depth)


        enhanced_opacity = opacity ** 0.2 * 1.0

        min_visible_opacity = 0.05
        if enhanced_opacity < min_visible_opacity and concentration_g_m2 > 0.3:
            enhanced_opacity = min_visible_opacity + opacity * 15  # Compensation factor

        return min(1.0, enhanced_opacity)

    def create_accumulation_zones(self):

        accumulation_map = np.zeros((self.height, self.width))
        num_centers = random.randint(6, 12)
        centers = []

        for _ in range(num_centers):
            center_x = random.randint(int(self.width * 0.1), int(self.width * 0.9))
            center_y = random.randint(int(self.height * 0.1), int(self.height * 0.9))
            intensity = random.uniform(0.5, 1.5)  # Increased intensity upper limit
            radius_x = random.randint(int(self.width * 0.1), int(self.width * 0.3))
            radius_y = random.randint(int(self.height * 0.1), int(self.height * 0.3))

            centers.append((center_x, center_y, intensity, radius_x, radius_y))
            y_indices, x_indices = np.ogrid[:self.height, :self.width]
            ellipse_mask = ((x_indices - center_x) / radius_x) ** 2 + ((y_indices - center_y) / radius_y) ** 2
            contribution = intensity * np.exp(-ellipse_mask * 1.2)
            accumulation_map += contribution

        noise = np.random.normal(0, 0.2, (self.height, self.width))  # Increased noise intensity
        accumulation_map += noise
        accumulation_map = gaussian_filter(accumulation_map, sigma=10)  # Reduced smoothing radius

        accumulation_map = np.clip(accumulation_map, 0, None)
        if accumulation_map.max() > 0:
            accumulation_map = accumulation_map / accumulation_map.max()

        return accumulation_map, centers

    def get_enhanced_dust_color(self, intensity_norm):

        if intensity_norm < 0.05:
            return (230, 225, 210)

        color_points = [
            (0.05, (40 / 360, 0.3, 0.9)),
            (0.2, (35 / 360, 0.4, 0.8)),
            (0.4, (30 / 360, 0.5, 0.7)),
            (0.6, (25 / 360, 0.6, 0.6)),
            (0.8, (20 / 360, 0.7, 0.5)),
            (1.0, (15 / 360, 0.8, 0.4))
        ]

        for i in range(len(color_points) - 1):
            t1, hsv1 = color_points[i]
            t2, hsv2 = color_points[i + 1]
            if t1 <= intensity_norm <= t2:
                ratio = (intensity_norm - t1) / (t2 - t1)
                h = hsv1[0] + (hsv2[0] - hsv1[0]) * ratio
                s = hsv1[1] + (hsv2[1] - hsv1[1]) * ratio
                v = hsv1[2] + (hsv2[2] - hsv1[2]) * ratio
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                return (int(r * 255), int(g * 255), int(b * 255))

        r, g, b = colorsys.hsv_to_rgb(15 / 360, 0.8, 0.4)
        return (int(r * 255), int(g * 255), int(b * 255))

    def create_enhanced_dust_layer(self, clean_img, target_concentration_g_m2):

        img = clean_img.copy().convert('RGBA')
        print(f"Generating dust at {target_concentration_g_m2:.1f} g/m² (Enhanced Visual Mode)")

        accumulation_map, centers = self.create_accumulation_zones()
        concentration_map = accumulation_map * target_concentration_g_m2 * 2.5

        intensity_map = np.zeros_like(concentration_map)
        for i in range(self.height):
            for j in range(self.width):
                local_conc = concentration_map[i, j]
                intensity_map[i, j] = self.enhanced_concentration_to_intensity(local_conc)

        print(f"✓ Concentration range: 0 - {np.max(concentration_map):.2f} g/m²")
        print(f"✓ Intensity range: 0 - {np.max(intensity_map):.3f}")
        print(f"✓ Number of accumulation centers: {len(centers)}")

        dust_overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(dust_overlay)
        block_size = 5
        for y in range(0, self.height, block_size):
            for x in range(0, self.width, block_size):
                y_end = min(y + block_size, self.height)
                x_end = min(x + block_size, self.width)
                block_intensity = np.mean(intensity_map[y:y_end, x:x_end])
                if block_intensity > 0.03:
                    color = self.get_enhanced_dust_color(block_intensity)
                    alpha = int(min(255, block_intensity * 300))
                    color_noise = random.randint(-15, 15)
                    final_color = tuple(max(0, min(255, c + color_noise)) for c in color)
                    alpha_noise = random.randint(-15, 15)
                    final_alpha = max(0, min(255, alpha + alpha_noise))
                    overlay_draw.rectangle([x, y, x_end, y_end], fill=(*final_color, final_alpha))

        dust_overlay = dust_overlay.filter(ImageFilter.GaussianBlur(radius=0.7))
        final_img = Image.alpha_composite(img, dust_overlay).convert('RGB')

        from PIL import ImageEnhance
        final_img = ImageEnhance.Brightness(final_img).enhance(0.93)
        final_img = ImageEnhance.Contrast(final_img).enhance(1.1)
        final_img = ImageEnhance.Sharpness(final_img).enhance(1.2)

        avg_concentration = np.mean(concentration_map[concentration_map > 0]) if np.any(concentration_map > 0) else 0
        coverage = np.sum(intensity_map > 0.03) / (self.width * self.height) * 100
        stats = {
            'target_concentration': target_concentration_g_m2,
            'avg_concentration': avg_concentration,
            'max_concentration': np.max(concentration_map),
            'coverage_percentage': coverage,
            'num_centers': len(centers),
            'intensity_range': (np.min(intensity_map), np.max(intensity_map))
        }

        print(f"✓ Average concentration: {avg_concentration:.2f} g/m², Coverage: {coverage:.1f}%")
        return final_img, stats

    def process_single_image(self, image_path, output_dir, num_samples=50, min_conc=0, max_conc=60):

        img_name = os.path.splitext(os.path.basename(image_path))[0]

        img_output_dir = os.path.join(output_dir, img_name)
        os.makedirs(img_output_dir, exist_ok=True)


        for item in os.listdir(img_output_dir):
            item_path = os.path.join(img_output_dir, item)


            if os.path.isfile(item_path):
                os.remove(item_path)


            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

        clean_img = self.load_image(image_path)


        concentrations = [random.uniform(min_conc, max_conc) for _ in range(num_samples)]
        print(
            f"Generating {num_samples} random concentration dust layers for {img_name} ({min_conc} - {max_conc} g/m²)")


        for i, conc in enumerate(concentrations):
            print(f"\n--- Generating dust at {conc:.2f} g/m² (Enhanced Mode) ---")
            dusty_img, stats = self.create_enhanced_dust_layer(clean_img, conc)
            filename = f"{i + 1:03d}_{conc:.2f}gm2.jpg"
            dusty_img.save(os.path.join(img_output_dir, filename))
            print(f"✓ Saved: {filename}")

        return img_output_dir

    def process_folder(self, input_folder, num_samples=50, min_conc=0, max_conc=60):


        if not os.path.isdir(input_folder):
            raise NotADirectoryError(f"Input path is not a valid folder: {input_folder}")

        output_dir = f"enhanced_red_dust_visualization"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Starting batch processing for folder: {input_folder}")
        print(f"All results will be saved to: {output_dir}")

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        for filename in os.listdir(input_folder):

            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(input_folder, filename)
                print(f"\n===== Starting to process image: {filename} =====")
                try:
                    self.process_single_image(image_path, output_dir, num_samples, min_conc, max_conc)
                    print(f"===== Image {filename} processed successfully =====")
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
                    continue

        return output_dir


def main():

    input_folder = "clean_pv_panel"

    try:

        simulator = EnhancedVisualDustSimulator()

        output_dir = simulator.process_folder(
            input_folder=input_folder,
            num_samples=50,
            min_conc=0,
            max_conc=60
        )
        print(f"\nAll images processed successfully! Results saved to: {output_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")




if __name__ == "__main__":
    main()
