import os
import json
import hashlib
import smtplib

from email.message import EmailMessage

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "anemia_model.keras"
)

FUND_REQUESTS_FILE = os.path.join(
    BASE_DIR,
    "backend",
    "fund_requests.json"
)

USERS_FILE = os.path.join(
    BASE_DIR,
    "backend",
    "users.json"
)

# Main .env location
ENV_FILE = os.path.join(
    BASE_DIR,
    ".env"
)

# Backup .env location
BACKEND_ENV_FILE = os.path.join(
    BASE_DIR,
    "backend",
    ".env"
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

# Load root .env first
if os.path.exists(ENV_FILE):
    load_dotenv(
        ENV_FILE,
        override=True
    )

# Also support backend/.env if present
if os.path.exists(BACKEND_ENV_FILE):
    load_dotenv(
        BACKEND_ENV_FILE,
        override=True
    )


EMAIL_ADDRESS = os.getenv(
    "EMAIL_ADDRESS",
    ""
).strip()

EMAIL_APP_PASSWORD = os.getenv(
    "EMAIL_APP_PASSWORD",
    ""
).strip()


# =========================================================
# ADMIN LOGIN CONFIGURATION
# =========================================================

ADMIN_EMAIL = "admin@synthera.com"

ADMIN_PASSWORD = "Admin@123"


# =========================================================
# EMAIL CONFIGURATION STATUS
# =========================================================

print()
print("==============================================")
print("        SYNTHERA EMAIL CONFIGURATION")
print("==============================================")

if EMAIL_ADDRESS:
    print(
        "Email address loaded:",
        EMAIL_ADDRESS
    )
else:
    print(
        "WARNING: EMAIL_ADDRESS is missing."
    )

if EMAIL_APP_PASSWORD:
    print(
        "Email app password loaded: YES"
    )
else:
    print(
        "WARNING: EMAIL_APP_PASSWORD is missing."
    )

print("==============================================")
print()


# =========================================================
# CREATE JSON FILES IF THEY DON'T EXIST
# =========================================================

if not os.path.exists(
    FUND_REQUESTS_FILE
):

    with open(
        FUND_REQUESTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            [],
            f,
            indent=4
        )


if not os.path.exists(
    USERS_FILE
):

    with open(
        USERS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            [],
            f,
            indent=4
        )


# =========================================================
# LOAD AI MODEL
# =========================================================

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "AI model loaded successfully."
    )

except Exception as e:

    model = None

    print(
        "Error loading AI model:",
        e
    )


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(

    title="Synthera AI Health Screening API",

    description=(
        "AI-Based Multi-Deficiency "
        "Health Screening & Fund Allocation System"
    ),

    version="1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5501",
        "http://localhost:5501"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {

        "message":
            "Welcome to Synthera AI Health Screening API",

        "status":
            "running"

    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model_loaded":
            model is not None,

        "email_configured":
            bool(
                EMAIL_ADDRESS
                and
                EMAIL_APP_PASSWORD
            )

    }


# =========================================================
# AI HEALTH SCREENING
# =========================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    allowed_types = [

        "image/jpeg",

        "image/png"

    ]

    if file.content_type not in allowed_types:

        raise HTTPException(

            status_code=400,

            detail=(
                "Only JPG, JPEG and PNG "
                "images are allowed."
            )

        )

    if model is None:

        raise HTTPException(

            status_code=500,

            detail="AI model is not loaded."

        )

    temp_file = os.path.join(

        BASE_DIR,

        "backend",

        "temp_screening_image"

    )

    try:

        contents = await file.read()

        with open(
            temp_file,
            "wb"
        ) as f:

            f.write(contents)

        image = Image.open(
            temp_file
        )

        image = image.convert(
            "RGB"
        )

        image = image.resize(
            (224, 224)
        )

        image_array = np.array(
            image
        )

        image_array = (
            image_array / 255.0
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        prediction = model.predict(
            image_array,
            verbose=0
        )

        probability = float(
            prediction[0][0]
        )

        if probability >= 0.5:

            result = "Anemic"

            confidence = (
                probability * 100
            )

        else:

            result = "Normal"

            confidence = (
                (1 - probability) * 100
            )

        return {

            "prediction":
                result,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "message": (
                "Screening result only. "
                "Please consult a qualified "
                "healthcare professional "
                "for medical evaluation."
            )

        }

    except UnidentifiedImageError:

        raise HTTPException(

            status_code=400,

            detail=(
                "The uploaded file "
                "is not a valid image."
            )

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Prediction failed: {str(e)}"
            )

        )

    finally:

        if os.path.exists(
            temp_file
        ):

            try:

                os.remove(
                    temp_file
                )

            except Exception:

                pass


# =========================================================
# FUND REQUEST MODEL
# =========================================================

class FundRequest(BaseModel):

    name: str

    email: str

    condition: str

    amount: float

    reason: str

    ai_result: str = ""

    ai_confidence: str = ""


# =========================================================
# CREATE FUND REQUEST
# =========================================================

@app.post("/fund-request")
def create_fund_request(
    request: FundRequest
):

    try:

        with open(
            FUND_REQUESTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            requests = json.load(f)

    except Exception:

        requests = []

    if requests:

        request_id = max(

            item.get(
                "id",
                0
            )

            for item in requests

        ) + 1

    else:

        request_id = 1

    ai_confidence = str(
        request.ai_confidence
    ).strip()

    if ai_confidence:

        if not ai_confidence.endswith("%"):

            ai_confidence += "%"

    new_request = {

        "id":
            request_id,

        "name":
            request.name.strip(),

        "email":
            request.email.strip(),

        "condition":
            request.condition.strip(),

        "amount":
            request.amount,

        "reason":
            request.reason.strip(),

        "ai_result":
            request.ai_result.strip(),

        "ai_confidence":
            ai_confidence,

        "status":
            "Pending",

        "allocated_amount":
            0

    }

    requests.append(
        new_request
    )

    with open(
        FUND_REQUESTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            requests,
            f,
            indent=4
        )

    return {

        "message":
            "Fund request submitted successfully.",

        "request":
            new_request

    }


# =========================================================
# GET ALL FUND REQUESTS
# =========================================================

@app.get("/fund-requests")
def get_fund_requests():

    try:

        with open(
            FUND_REQUESTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            requests = json.load(f)

    except Exception:

        requests = []

    return requests


# =========================================================
# SEND ALLOCATION EMAIL
# =========================================================

def send_allocation_email(

    recipient_email,

    recipient_name,

    request_id,

    allocated_amount

):

    print()
    print("----------------------------------------------")
    print("Attempting to send allocation email...")
    print("Recipient:", recipient_email)
    print("Request ID:", request_id)
    print("Amount:", allocated_amount)
    print("----------------------------------------------")

    if not EMAIL_ADDRESS:

        print(
            "EMAIL_ADDRESS is missing."
        )

        return {

            "success":
                False,

            "message":
                "EMAIL_ADDRESS is missing."

        }

    if not EMAIL_APP_PASSWORD:

        print(
            "EMAIL_APP_PASSWORD is missing."
        )

        return {

            "success":
                False,

            "message":
                "EMAIL_APP_PASSWORD is missing."

        }

    try:

        message = EmailMessage()

        message["Subject"] = (
            "Synthera - Fund Request Allocated"
        )

        message["From"] = (
            EMAIL_ADDRESS
        )

        message["To"] = (
            recipient_email
        )

        email_body = f"""
Dear {recipient_name},

Your fund request has been successfully
allocated through Synthera.

Request ID: {request_id}

Allocated Amount: ₹{allocated_amount:,.2f}

Your request has been reviewed and the
above amount has been allocated.

Thank you for using Synthera.

Regards,
Synthera Team

AI Health Screening & Fund Allocation System
"""

        message.set_content(
            email_body
        )

        print(
            "Connecting to Gmail SMTP..."
        )

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            timeout=30
        ) as server:

            server.login(
                EMAIL_ADDRESS,
                EMAIL_APP_PASSWORD
            )

            print(
                "Gmail login successful."
            )

            server.send_message(
                message
            )

        print(
            "EMAIL SENT SUCCESSFULLY."
        )

        return {

            "success":
                True,

            "message":
                "Email notification sent successfully."

        }

    except Exception as e:

        error_message = str(e)

        print(
            "EMAIL SENDING FAILED:"
        )

        print(
            error_message
        )

        return {

            "success":
                False,

            "message":
                f"Email sending failed: {error_message}"

        }


# =========================================================
# SEND REJECTION EMAIL
# =========================================================

def send_rejection_email(

    recipient_email,

    recipient_name,

    request_id

):

    if not EMAIL_ADDRESS:

        return {

            "success":
                False,

            "message":
                "EMAIL_ADDRESS is missing."

        }

    if not EMAIL_APP_PASSWORD:

        return {

            "success":
                False,

            "message":
                "EMAIL_APP_PASSWORD is missing."

        }

    try:

        message = EmailMessage()

        message["Subject"] = (
            "Synthera - Fund Request Update"
        )

        message["From"] = (
            EMAIL_ADDRESS
        )

        message["To"] = (
            recipient_email
        )

        email_body = f"""
Dear {recipient_name},

Your fund request has been reviewed by the
Synthera admin team.

Request ID: {request_id}

Status: Rejected

We are sorry to inform you that your financial
support request could not be approved at this time.

Thank you for using Synthera.

Regards,
Synthera Team

AI Health Screening & Fund Allocation System
"""

        message.set_content(
            email_body
        )

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            timeout=30
        ) as server:

            server.login(
                EMAIL_ADDRESS,
                EMAIL_APP_PASSWORD
            )

            server.send_message(
                message
            )

        print(
            "REJECTION EMAIL SENT SUCCESSFULLY."
        )

        return {

            "success":
                True,

            "message":
                "Rejection email sent successfully."

        }

    except Exception as e:

        error_message = str(e)

        print(
            "REJECTION EMAIL FAILED:",
            error_message
        )

        return {

            "success":
                False,

            "message":
                f"Rejection email sending failed: {error_message}"

        }


# =========================================================
# ALLOCATION MODEL
# =========================================================

class AllocationRequest(BaseModel):

    allocated_amount: float


# =========================================================
# ALLOCATE FUND
# =========================================================

@app.put(
    "/allocate-fund/{request_id}"
)
def allocate_fund(

    request_id: int,

    allocation: AllocationRequest

):

    if allocation.allocated_amount < 0:

        raise HTTPException(

            status_code=400,

            detail=(
                "Allocated amount "
                "cannot be negative."
            )

        )

    try:

        with open(
            FUND_REQUESTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            requests = json.load(f)

    except Exception:

        raise HTTPException(

            status_code=500,

            detail=(
                "Unable to read fund requests."
            )

        )

    selected_request = None

    for request in requests:

        if request.get(
            "id"
        ) == request_id:

            selected_request = request

            break

    if selected_request is None:

        raise HTTPException(

            status_code=404,

            detail=(
                "Fund request not found."
            )

        )

    selected_request[
        "status"
    ] = "Allocated"

    selected_request[
        "allocated_amount"
    ] = allocation.allocated_amount

    with open(
        FUND_REQUESTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            requests,
            f,
            indent=4
        )

    # -----------------------------------------------------
    # SEND EMAIL
    # -----------------------------------------------------

    email_result = send_allocation_email(

        selected_request["email"],

        selected_request["name"],

        selected_request["id"],

        allocation.allocated_amount

    )

    return {

        "message":
            "Fund allocated successfully.",

        "request":
            selected_request,

        "email_status":
            email_result

    }


# =========================================================
# REJECT FUND
# =========================================================

@app.put(
    "/reject-fund/{request_id}"
)
def reject_fund(

    request_id: int

):

    try:

        with open(
            FUND_REQUESTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            requests = json.load(f)

    except Exception:

        requests = []

    selected_request = None

    for request in requests:

        if request.get(
            "id"
        ) == request_id:

            selected_request = request

            break

    if selected_request is None:

        raise HTTPException(

            status_code=404,

            detail="Fund request not found."

        )

    if (
        str(
            selected_request.get(
                "status",
                ""
            )
        ).lower()
        == "allocated"
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "An allocated request "
                "cannot be rejected."
            )

        )

    selected_request[
        "status"
    ] = "Rejected"

    selected_request[
        "allocated_amount"
    ] = 0

    with open(
        FUND_REQUESTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            requests,
            f,
            indent=4
        )

    email_result = send_rejection_email(

        selected_request["email"],

        selected_request["name"],

        selected_request["id"]

    )

    return {

        "message":
            "Fund request rejected successfully.",

        "request":
            selected_request,

        "email_status":
            email_result

    }


# =========================================================
# SIGN UP MODEL
# =========================================================

class SignupRequest(BaseModel):

    name: str

    email: str

    password: str


# =========================================================
# SIGN UP
# =========================================================

@app.post("/signup")
def signup(

    user: SignupRequest

):

    name = user.name.strip()

    email = user.email.strip().lower()

    password = user.password

    if not name:

        raise HTTPException(

            status_code=400,

            detail="Name is required."

        )

    if not email:

        raise HTTPException(

            status_code=400,

            detail="Email is required."

        )

    if len(password) < 6:

        raise HTTPException(

            status_code=400,

            detail=(
                "Password must contain "
                "at least 6 characters."
            )

        )

    try:

        with open(
            USERS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            users = json.load(f)

    except Exception:

        users = []

    for existing_user in users:

        if existing_user.get(
            "email"
        ) == email:

            raise HTTPException(

                status_code=400,

                detail="Email already registered."

            )

    password_hash = hashlib.sha256(

        password.encode(
            "utf-8"
        )

    ).hexdigest()

    if users:

        user_id = max(

            item.get(
                "id",
                0
            )

            for item in users

        ) + 1

    else:

        user_id = 1

    new_user = {

        "id":
            user_id,

        "name":
            name,

        "email":
            email,

        "password_hash":
            password_hash

    }

    users.append(
        new_user
    )

    with open(
        USERS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            users,
            f,
            indent=4
        )

    return {

        "message":
            "Account created successfully.",

        "user_id":
            user_id,

        "name":
            name,

        "email":
            email

    }


# =========================================================
# LOGIN MODEL
# =========================================================

class LoginRequest(BaseModel):

    email: str

    password: str


# =========================================================
# LOGIN
# =========================================================

@app.post("/login")
def login(

    user: LoginRequest

):

    email = (
        user.email
        .strip()
        .lower()
    )

    password = user.password

    if not email:

        raise HTTPException(

            status_code=400,

            detail="Email is required."

        )

    if not password:

        raise HTTPException(

            status_code=400,

            detail="Password is required."

        )

    try:

        with open(
            USERS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            users = json.load(f)

    except Exception:

        users = []

    password_hash = hashlib.sha256(

        password.encode(
            "utf-8"
        )

    ).hexdigest()

    for existing_user in users:

        if existing_user.get(
            "email"
        ) == email:

            if (
                existing_user.get(
                    "password_hash"
                )
                != password_hash
            ):

                raise HTTPException(

                    status_code=401,

                    detail="Incorrect password."

                )

            return {

                "message":
                    "Login successful.",

                "user": {

                    "id":
                        existing_user.get(
                            "id"
                        ),

                    "name":
                        existing_user.get(
                            "name"
                        ),

                    "email":
                        existing_user.get(
                            "email"
                        )

                }

            }

    raise HTTPException(

        status_code=401,

        detail=(
            "No account found "
            "with this email."
        )

    )


# =========================================================
# ADMIN LOGIN MODEL
# =========================================================

class AdminLoginRequest(BaseModel):

    email: str

    password: str


# =========================================================
# ADMIN LOGIN
# =========================================================

@app.post("/admin-login")
def admin_login(

    admin: AdminLoginRequest

):

    email = (
        admin.email
        .strip()
        .lower()
    )

    password = admin.password

    if not email:

        raise HTTPException(

            status_code=400,

            detail="Admin email is required."

        )

    if not password:

        raise HTTPException(

            status_code=400,

            detail="Admin password is required."

        )

    if (
        email == ADMIN_EMAIL.lower()
        and
        password == ADMIN_PASSWORD
    ):

        return {

            "message":
                "Admin login successful.",

            "admin": {

                "email":
                    ADMIN_EMAIL,

                "role":
                    "admin"

            }

        }

    raise HTTPException(

        status_code=401,

        detail="Invalid admin email or password."

    )


# =========================================================
# SERVER START MESSAGE
# =========================================================

print()

print(
    "=============================================="
)

print(
    "        SYNTHERA BACKEND STARTED"
)

print(
    "=============================================="
)

print(
    "AI Screening      : /predict"
)

print(
    "Fund Request      : /fund-request"
)

print(
    "Admin Requests    : /fund-requests"
)

print(
    "Fund Allocation   : /allocate-fund/{id}"
)

print(
    "Fund Rejection    : /reject-fund/{id}"
)

print(
    "User Sign Up      : /signup"
)

print(
    "User Login        : /login"
)

print(
    "Admin Login       : /admin-login"
)

print(
    "Email Configured  :",
    bool(
        EMAIL_ADDRESS
        and
        EMAIL_APP_PASSWORD
    )
)

print(
    "=============================================="
)

print()