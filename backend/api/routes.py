from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Form
)

import os
import uuid
from pathlib import Path

from backend.vision.pose_analyzer import PoseAnalyzer

from backend.features.badminton_features import (
    extract_badminton_features
)

from backend.sports.badminton import (
    analyze_badminton
)

from backend.services.analysis_service import (
    analyze_football_video
)

from backend.services.athletics_service import (
    analyze_athletics_video
)

from backend.services.basketball_service import (
    analyze_basketball_video
)

from backend.services.progress_service import (
    save_performance_result,
    get_user_history,
    get_progress_data
)


router = APIRouter()

UPLOAD_DIR = str(Path(__file__).resolve().parents[2] / "uploads")

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


MAX_UPLOAD_SIZE = 200 * 1024 * 1024


# =========================================================
# VIDEO VALIDATION
# =========================================================

def validate_video_extension(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MP4, AVI, MOV or MKV."
            )
        )

    return extension


# =========================================================
# SAVE UPLOADED VIDEO
# =========================================================

async def save_upload_file(
    file,
    destination
):

    total_size = 0

    with open(
        destination,
        "wb"
    ) as buffer:

        while True:

            chunk = await file.read(
                1024 * 1024
            )

            if not chunk:
                break

            total_size += len(chunk)

            if total_size > MAX_UPLOAD_SIZE:

                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Video is too large. "
                        "Maximum allowed size is 200 MB."
                    )
                )

            buffer.write(
                chunk
            )


# =========================================================
# BASIC VIDEO UPLOAD
# =========================================================

@router.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...)
):

    extension = validate_video_extension(
        file.filename
    )

    safe_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    try:

        await save_upload_file(
            file,
            file_path
        )

    except HTTPException:

        if os.path.exists(
            file_path
        ):
            os.remove(
                file_path
            )

        raise

    except Exception as error:

        if os.path.exists(
            file_path
        ):
            os.remove(
                file_path
            )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not save video: {error}"
            )
        )

    return {
        "success": True,
        "message": "Video uploaded successfully",
        "original_filename": file.filename,
        "stored_filename": safe_filename
    }


# =========================================================
# BADMINTON ANALYSIS + DATABASE SAVE
# =========================================================

@router.post("/analyze/badminton")
async def analyze_badminton_video(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):

    extension = validate_video_extension(
        file.filename
    )

    safe_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    analyzer = None

    try:

        await save_upload_file(
            file,
            file_path
        )

        analyzer = PoseAnalyzer(
            target_fps=5
        )

        pose_result = analyzer.analyze_video(
            file_path
        )

        if pose_result[
            "processed_frames"
        ] < 3:

            raise HTTPException(
                status_code=422,
                detail=(
                    "The video does not contain "
                    "enough usable frames."
                )
            )

        if pose_result[
            "pose_detected_frames"
        ] == 0:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No usable human pose was "
                    "detected in the video."
                )
            )

        detection_rate = pose_result[
            "detection_rate"
        ]

        if detection_rate < 0.30:

            raise HTTPException(
                status_code=422,
                detail=(
                    "Pose detection quality is too low. "
                    "Use a clearer video where the player "
                    "is visible."
                )
            )

        features = extract_badminton_features(
            pose_result["landmarks"]
        )

        analysis = analyze_badminton(
            features
        )

        analysis["pose_quality"] = {

            "total_video_frames":
                pose_result["total_frames"],

            "processed_frames":
                pose_result["processed_frames"],

            "pose_detected_frames":
                pose_result["pose_detected_frames"],

            "detection_rate":
                pose_result["detection_rate"],

            "original_fps":
                pose_result["original_fps"],

            "analysis_fps":
                pose_result["target_fps"]
        }

        save_result = save_performance_result(
            user_id=user_id,
            sport="badminton",
            analysis_result=analysis
        )

        analysis["saved_performance"] = (
            save_result
        )

        return analysis

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed: {error}"
            )
        )

    finally:

        if analyzer is not None:
            analyzer.close()

        if os.path.exists(
            file_path
        ):
            os.remove(
                file_path
            )


# =========================================================
# FOOTBALL ANALYSIS + DATABASE SAVE
# =========================================================

@router.post("/analyze/football")
async def analyze_football_endpoint(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):

    extension = validate_video_extension(
        file.filename
    )

    safe_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    try:

        await save_upload_file(
            file,
            file_path
        )

        analysis = analyze_football_video(
            file_path,
            target_fps=5
        )

        save_result = save_performance_result(
            user_id=user_id,
            sport="football",
            analysis_result=analysis
        )

        analysis["saved_performance"] = (
            save_result
        )

        return analysis

    except HTTPException:

        raise

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Football analysis failed: {error}"
            )
        )

    finally:

        if os.path.exists(
            file_path
        ):
            os.remove(
                file_path
            )


# =========================================================
# ATHLETICS SPRINT ANALYSIS + DATABASE SAVE
# =========================================================

@router.post("/analyze/athletics")
async def analyze_athletics_endpoint(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):

    extension = validate_video_extension(
        file.filename
    )

    safe_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    try:

        await save_upload_file(
            file,
            file_path
        )

        analysis = analyze_athletics_video(
            file_path,
            target_fps=5
        )

        save_result = save_performance_result(
            user_id=user_id,
            sport="athletics",
            analysis_result=analysis
        )

        analysis["saved_performance"] = (
            save_result
        )

        return analysis

    except HTTPException:

        raise

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Athletics analysis failed: {error}"
            )
        )

    finally:

        if os.path.exists(
            file_path
        ):
            os.remove(
                file_path
            )


# =========================================================
# BASKETBALL ANALYSIS + DATABASE SAVE
# =========================================================

@router.post("/analyze/basketball")
async def analyze_basketball_endpoint(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):

    extension = validate_video_extension(
        file.filename
    )

    safe_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    try:

        await save_upload_file(
            file,
            file_path
        )

        analysis = analyze_basketball_video(
            file_path,
            target_fps=5
        )

        save_result = save_performance_result(
            user_id=user_id,
            sport="basketball",
            analysis_result=analysis
        )

        analysis["saved_performance"] = (
            save_result
        )

        return analysis

    except HTTPException:

        raise

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Basketball analysis failed: {error}"
            )
        )

    finally:

        if os.path.exists(
            file_path
        ):
            os.remove(
                file_path
            )


# =========================================================
# USER HISTORY
# =========================================================

@router.get("/history/{user_id}")
def user_history(
    user_id: str
):

    history = get_user_history(
        user_id=user_id
    )

    return {
        "user_id": user_id,
        "total_records": len(
            history
        ),
        "history": history
    }


# =========================================================
# SPORT-SPECIFIC HISTORY
# =========================================================

@router.get(
    "/history/{user_id}/{sport}"
)
def user_sport_history(
    user_id: str,
    sport: str
):

    history = get_user_history(
        user_id=user_id,
        sport=sport
    )

    return {
        "user_id": user_id,
        "sport": sport,
        "total_records": len(
            history
        ),
        "history": history
    }


# =========================================================
# GRAPH-READY PROGRESS DATA
# =========================================================

@router.get(
    "/progress/{user_id}/{sport}"
)
def user_progress(
    user_id: str,
    sport: str
):

    progress_data = get_progress_data(
        user_id=user_id,
        sport=sport
    )

    return progress_data