import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from AdaAttN.constants import DEVICE

user_pref = "equal"

weightages = {
    "equal": [0.033, 0.34, 0.33],
    "stylish": [0.02, 0.4, 0.4],
    "similar": [0.03, 0.5, 0.2]
}

lambda_pixel, lambda_structure, lambda_semantic = weightages[user_pref]

def calc_ssim(content_img_tensor, stylized_img_tensor):
    ssim_obj = StructuralSimilarityIndexMeasure(data_range=255.0).to(DEVICE)
    ssim_result = ssim_obj(stylized_img_tensor, content_img_tensor)

    return ssim_result

def calc_psnr(content_img_tensor, stylized_img_tensor):
    psnr_obj = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    result = psnr_obj(stylized_img_tensor, content_img_tensor)
    return result

def calc_iag():
    pass

def calc_effectiveness():
    pass

def calc_color_similarity_score():
    pass

def calc_lpips(content_img_tensor, stylized_img_tensor):
    vgg_lpips = lpips.LPIPS(net='vgg').to(DEVICE)

    lpips_result = vgg_lpips(content_img_tensor, stylized_img_tensor)

    return lpips_result

def calc_final_score(content_img_tensor, stylized_img_tensor):
    pixel_results =  lambda_pixel *calc_psnr(content_img_tensor, stylized_img_tensor)
    structural_results =  lambda_structure * (calc_lpips(content_img_tensor, stylized_img_tensor) 
                            + calc_ssim(content_img_tensor, stylized_img_tensor))

    final_score =  pixel_results +  structural_results

    return {
        "Pixel_score": pixel_results,
        "Structural_score": structural_results,
        "Final_score": final_score
    }
    