import pytest
import uuid


@pytest.mark.asyncio
async def test_health_endpoints(client):
    res = await client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_user_registration_and_login(client):
    unique_email = f"farmer_{uuid.uuid4().hex[:8]}@example.com"
    reg_payload = {
        "email": unique_email,
        "password": "SecurePassword123!",
        "full_name": "Ramesh Patil",
        "role": "farmer",
    }

    # 1. Register new user
    reg_res = await client.post("/api/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    user_data = reg_res.json()
    assert user_data["email"] == unique_email
    assert user_data["full_name"] == "Ramesh Patil"
    assert user_data["role"] == "farmer"
    assert "id" in user_data

    # 2. Duplicate registration must return HTTP 409 CONFLICT
    dup_res = await client.post("/api/auth/register", json=reg_payload)
    assert dup_res.status_code == 409
    dup_data = dup_res.json()
    assert "error" in dup_data
    assert "reason" in dup_data
    assert "already exists" in dup_data["reason"].lower()

    # 3. Login with correct credentials
    login_res = await client.post(
        "/api/auth/login",
        json={"email": unique_email, "password": "SecurePassword123!"},
    )
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    token = token_data["access_token"]

    # 4. Login with incorrect password should fail with 401
    wrong_login = await client.post(
        "/api/auth/login",
        json={"email": unique_email, "password": "WrongPassword"},
    )
    assert wrong_login.status_code == 401
    assert "reason" in wrong_login.json()

    # 5. Fetch profile using /api/auth/me
    me_res = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == unique_email
    assert me_data["full_name"] == "Ramesh Patil"


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    email = f"dup_{uuid.uuid4().hex[:8]}@example.com"
    payload = {
        "email": email,
        "password": "ValidPassword123",
        "full_name": "Test Farmer",
        "role": "farmer",
    }
    # First registration
    res1 = await client.post("/api/auth/register", json=payload)
    assert res1.status_code == 201

    # Second registration with same email
    res2 = await client.post("/api/auth/register", json=payload)
    assert res2.status_code == 409
    assert res2.json()["error"] == "HTTP_409"


@pytest.mark.asyncio
async def test_register_invalid_password(client):
    # Password too short (< 6 characters)
    short_pwd_res = await client.post(
        "/api/auth/register",
        json={
            "email": f"short_{uuid.uuid4().hex[:8]}@example.com",
            "password": "123",
            "full_name": "Short Pwd",
            "role": "farmer",
        },
    )
    assert short_pwd_res.status_code == 422
    assert "reason" in short_pwd_res.json()

    # Password too long (> 72 characters)
    long_pwd_res = await client.post(
        "/api/auth/register",
        json={
            "email": f"long_{uuid.uuid4().hex[:8]}@example.com",
            "password": "A" * 75,
            "full_name": "Long Pwd",
            "role": "farmer",
        },
    )
    assert long_pwd_res.status_code == 422
    assert "reason" in long_pwd_res.json()


@pytest.mark.asyncio
async def test_register_invalid_email(client):
    res = await client.post(
        "/api/auth/register",
        json={
            "email": "not-an-email",
            "password": "ValidPassword123",
            "full_name": "Invalid Email",
            "role": "farmer",
        },
    )
    assert res.status_code == 422
    assert "reason" in res.json()


@pytest.mark.asyncio
async def test_register_invalid_role(client):
    res = await client.post(
        "/api/auth/register",
        json={
            "email": f"role_{uuid.uuid4().hex[:8]}@example.com",
            "password": "ValidPassword123",
            "full_name": "Invalid Role",
            "role": "astronaut",
        },
    )
    assert res.status_code == 422
    assert "reason" in res.json()


@pytest.mark.asyncio
async def test_protected_route_unauthorized(client):
    res = await client.get("/api/auth/me")
    assert res.status_code == 401
    data = res.json()
    assert "error" in data
    assert "reason" in data
