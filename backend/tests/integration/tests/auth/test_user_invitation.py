import pytest
from requests import HTTPError

from onyx.auth.schemas import UserRole
from tests.integration.common_utils.managers.user import UserManager
from tests.integration.common_utils.test_models import DATestUser


def test_inviting_users_flow(reset: None) -> None:
    """
    Test that verifies the functionality around inviting users:
      1. Creating an admin user
      2. Admin inviting a new user
      3. Invited user successfully signing in
      4. Non-invited user attempting to sign in (should result in an error)
    """
    # 1) Create an admin user (the first user created is automatically admin)
    admin_user: DATestUser = UserManager.create(name="admin_user")
    assert admin_user is not None
    assert UserManager.is_role(admin_user, UserRole.ADMIN)

    # 2) Admin invites a new user
    invited_email = "invited_user@test.com"
    invite_response = UserManager.invite_users(admin_user, [invited_email])

    assert invite_response == 1

    # 3) The invited user successfully registers/logs in
    invited_user: DATestUser = UserManager.create(
        name="invited_user", email=invited_email
    )
    assert invited_user is not None
    assert invited_user.email == invited_email
    assert UserManager.is_role(invited_user, UserRole.BASIC)

    # 4) A non-invited user attempts to sign in/register (should fail)
    with pytest.raises(HTTPError):
        UserManager.create(name="uninvited_user", email="uninvited_user@test.com")
