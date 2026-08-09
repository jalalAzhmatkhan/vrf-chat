import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LoginPage } from './LoginPage';
import { AuthProvider } from '../../auth/AuthContext';
import { ApiError } from '../../lib/apiError';
import * as authApi from '../../auth/authApi';

vi.mock('../../auth/authApi');

function renderLoginPage() {
  return render(
    <MemoryRouter initialEntries={['/login']}>
      <AuthProvider>
        <LoginPage />
      </AuthProvider>
    </MemoryRouter>,
  );
}

describe('LoginPage', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('disables the submit button until both fields are filled', async () => {
    const user = userEvent.setup();
    renderLoginPage();

    const submitButton = screen.getByRole('button', { name: 'Masuk' });
    expect(submitButton).toBeDisabled();

    await user.type(screen.getByLabelText('Username'), 'jalal');
    expect(submitButton).toBeDisabled();

    await user.type(screen.getByLabelText('Password'), 'super-secret');
    expect(submitButton).toBeEnabled();
  });

  it('shows the 401 alert, clears the password field, and does not navigate away', async () => {
    authApi.login.mockRejectedValue(new ApiError({ status: 401, detail: 'Invalid username or password' }));
    const user = userEvent.setup();
    renderLoginPage();

    await user.type(screen.getByLabelText('Username'), 'jalal');
    await user.type(screen.getByLabelText('Password'), 'wrong-password');
    await user.click(screen.getByRole('button', { name: 'Masuk' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Username atau password salah.');
    expect(screen.getByLabelText('Password')).toHaveValue('');
    expect(screen.getByLabelText('Username')).toHaveValue('jalal');
  });

  it('shows a mono countdown on 429 using retry_after_seconds from the backend', async () => {
    authApi.login.mockRejectedValue(
      new ApiError({ status: 429, detail: 'Too many login attempts, try again later', retryAfterSeconds: 47 }),
    );
    const user = userEvent.setup();
    renderLoginPage();

    await user.type(screen.getByLabelText('Username'), 'jalal');
    await user.type(screen.getByLabelText('Password'), 'whatever');
    await user.click(screen.getByRole('button', { name: 'Masuk' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('0:47');
    expect(screen.getByRole('button', { name: 'Masuk' })).toBeDisabled();
    expect(screen.getByLabelText('Username')).toBeDisabled();
  });

  it('shows a network error message and re-enables the button on network failure', async () => {
    authApi.login.mockRejectedValue(new ApiError({ status: 'network', detail: 'Network request failed' }));
    const user = userEvent.setup();
    renderLoginPage();

    await user.type(screen.getByLabelText('Username'), 'jalal');
    await user.type(screen.getByLabelText('Password'), 'whatever');
    await user.click(screen.getByRole('button', { name: 'Masuk' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Tidak dapat terhubung ke server');
    expect(screen.getByRole('button', { name: 'Masuk' })).toBeEnabled();
  });

  it('calls login and GET /auth/me on successful submit', async () => {
    authApi.login.mockResolvedValue({ access_token: 'jwt', token_type: 'bearer', expires_in: 1800 });
    authApi.me.mockResolvedValue({ id: 1, username: 'jalal', role: 'admin', scopes: ['chat:read', 'chat:write'] });
    const user = userEvent.setup();
    renderLoginPage();

    await user.type(screen.getByLabelText('Username'), 'jalal');
    await user.type(screen.getByLabelText('Password'), 'correct-password');
    await user.click(screen.getByRole('button', { name: 'Masuk' }));

    await waitFor(() => expect(authApi.me).toHaveBeenCalled());
    expect(authApi.login).toHaveBeenCalledWith('jalal', 'correct-password');
  });
});
