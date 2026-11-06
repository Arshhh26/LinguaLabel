"""
LinguaLabel — Authentication UI Views (Part 11)

Provides secure login, registration, and logout interfaces with Argon2id password
hashing, input validation, generic failure messages, and session token generation.
"""

from typing import Optional
import streamlit as st

from db.auth import (
    authenticate_user,
    create_session_token,
    register_user,
    validate_registration_inputs,
    verify_session_token,
)
from db.models import User
from db.service import list_workspaces
from db.session import get_db_session


def render_auth_page() -> None:
    """Renders the login and registration tabs."""
    st.markdown("""
    <div style="max-width: 480px; margin: 2rem auto; text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌐</div>
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; margin-bottom: 0.25rem;">
            LinguaLabel
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 2rem;">
            Secure Multilingual NLP Annotation & Personal Workspaces
        </p>
    </div>
    """, unsafe_allow_html=True)

    auth_card = st.container()
    with auth_card:
        c1, c2, c3 = st.columns([1, 2.2, 1])
        with c2:
            tab_login, tab_register = st.tabs(["🔐 Sign In", "✨ Create Account"])

            # -------------------------------------------------------------
            # TAB 1: LOGIN
            # -------------------------------------------------------------
            with tab_login:
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                login_ident = st.text_input(
                    "Email or Username",
                    key="login_ident",
                    placeholder="name@example.com or username",
                )
                login_pass = st.text_input(
                    "Password",
                    type="password",
                    key="login_pass",
                    placeholder="Enter your password",
                )

                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                if st.button("Sign In to LinguaLabel →", use_container_width=True, type="primary"):
                    if not login_ident or not login_pass:
                        st.error("Please enter both your email/username and password.")
                    else:
                        with st.spinner("Authenticating securely..."):
                            user, err = authenticate_user(login_ident, login_pass)
                            if err or not user:
                                st.error(err or "Invalid email or password.")
                            else:
                                # Set session state
                                token = create_session_token(user.id)
                                st.session_state.auth_token = token
                                st.session_state.current_user = user.to_dict()
                                st.session_state.msg = f"Welcome back, {user.username}!"
                                st.rerun()

            # -------------------------------------------------------------
            # TAB 2: REGISTER
            # -------------------------------------------------------------
            with tab_register:
                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                reg_email = st.text_input(
                    "Email Address",
                    key="reg_email",
                    placeholder="you@example.com",
                )
                reg_username = st.text_input(
                    "Username",
                    key="reg_username",
                    placeholder="letters, numbers, underscore (3-30 chars)",
                )
                reg_pass1 = st.text_input(
                    "Password (min 8 chars, 1 letter, 1 number)",
                    type="password",
                    key="reg_pass1",
                    placeholder="Create a strong password",
                )
                reg_pass2 = st.text_input(
                    "Confirm Password",
                    type="password",
                    key="reg_pass2",
                    placeholder="Re-enter your password",
                )

                st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                if st.button("Create Personal Account ✨", use_container_width=True, type="primary"):
                    if reg_pass1 != reg_pass2:
                        st.error("Passwords do not match.")
                    else:
                        valid, v_err = validate_registration_inputs(reg_email, reg_username, reg_pass1)
                        if not valid:
                            st.error(v_err)
                        else:
                            with st.spinner("Creating your isolated workspace..."):
                                user, err = register_user(reg_email, reg_username, reg_pass1)
                                if err or not user:
                                    st.error(err or "Failed to register account.")
                                else:
                                    token = create_session_token(user.id)
                                    st.session_state.auth_token = token
                                    st.session_state.current_user = user.to_dict()
                                    st.session_state.msg = f"Account created! Welcome to LinguaLabel, {user.username}."
                                    st.rerun()


def logout_current_user() -> None:
    """Clears authentication session state and redirects to login."""
    keys_to_clear = [
        "auth_token",
        "current_user",
        "active_workspace_id",
        "active_dataset_id",
        "active_df",
        "active_dataset_key",
        "active_storage_path",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

