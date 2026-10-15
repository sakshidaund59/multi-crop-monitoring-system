import pytest
import uuid


@pytest.mark.asyncio
async def test_farm_crud_flow(client):
    unique_email = f"farmer_geo_{uuid.uuid4().hex[:8]}@example.com"
    reg_payload = {
        "email": unique_email,
        "password": "FarmPassword123!",
        "full_name": "Suresh Deshmukh",
        "role": "farmer",
    }

    # 1. Register & login to get token
    await client.post("/api/auth/register", json=reg_payload)
    login_res = await client.post(
        "/api/auth/login",
        json={"email": unique_email, "password": "FarmPassword123!"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Initially list farms -> empty list
    list_res = await client.get("/api/farms", headers=headers)
    assert list_res.status_code == 200
    assert list_res.json() == []

    # 3. Create a farm with GPS point and boundary polygon
    farm_payload = {
        "name": "Green Valley Farm",
        "latitude": 19.8762,
        "longitude": 75.3433,
        "area_ha": 12.5,
        "boundary": {
            "type": "Polygon",
            "coordinates": [
                [
                    [75.3400, 19.8750],
                    [75.3460, 19.8750],
                    [75.3460, 19.8800],
                    [75.3400, 19.8800],
                    [75.3400, 19.8750],
                ]
            ],
        },
    }
    create_res = await client.post("/api/farms", json=farm_payload, headers=headers)
    assert create_res.status_code == 201
    farm_data = create_res.json()
    assert farm_data["name"] == "Green Valley Farm"
    assert farm_data["latitude"] == 19.8762
    assert farm_data["longitude"] == 75.3433
    assert farm_data["area_ha"] == 12.5
    assert farm_data["boundary"] is not None
    assert "id" in farm_data
    farm_id = farm_data["id"]

    # 4. Get specific farm by ID
    get_res = await client.get(f"/api/farms/{farm_id}", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["id"] == farm_id

    # 5. List farms -> contains 1 farm
    list_after = await client.get("/api/farms", headers=headers)
    assert list_after.status_code == 200
    assert len(list_after.json()) == 1

    # 6. Delete farm
    del_res = await client.delete(f"/api/farms/{farm_id}", headers=headers)
    assert del_res.status_code == 204

    # 7. Get deleted farm -> 404
    get_del = await client.get(f"/api/farms/{farm_id}", headers=headers)
    assert get_del.status_code == 404


@pytest.mark.asyncio
async def test_farm_unauthorized_access(client):
    res = await client.get("/api/farms")
    assert res.status_code == 401
    assert "error" in res.json()
    assert "reason" in res.json()
