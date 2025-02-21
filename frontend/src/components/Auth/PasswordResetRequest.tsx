import React, { useState } from 'react';
import api from '../../api';

const PasswordResetRequest: React.FC = () => {
    const [email, setEmail] = useState('');
    const [message, setMessage] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            await api.post('/users/password-reset-request', { email });
            setMessage('Check your email for reset instructions');
        } catch (error) {
            setMessage('Error requesting password reset');
        }
    };

    return (
        <div className="password-reset">
            <h2>Reset Password</h2>
            <form onSubmit={handleSubmit}>
                <input
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="Enter your email"
                    required
                />
                <button type="submit">Send Reset Link</button>
            </form>
            {message && <p>{message}</p>}
        </div>
    );
};

export default PasswordResetRequest; 