import React, { useState } from 'react';
import { User } from '../../types';
import api from '../../api';

interface UserFormProps {
    user?: User;
    onSuccess: () => void;
}

interface UserFormState {
    username: string;
    email: string;
    roles: string;
    password: string;
}

const UserForm: React.FC<UserFormProps> = ({ user, onSuccess }) => {
    const [formData, setFormData] = useState<UserFormState>({
        username: user?.username || '',
        email: user?.email || '',
        roles: user?.roles.join(', ') || '',
        password: ''
    });

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const userData = {
                ...formData,
                roles: formData.roles.split(',').map(r => r.trim())
            };
            
            if (user) {
                await api.put(`/users/${user.username}`, userData);
            } else {
                await api.post('/users', userData);
            }
            onSuccess();
        } catch (error) {
            console.error('Error saving user:', error);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <div>
                <label>Username:</label>
                <input 
                    type="text" 
                    value={formData.username}
                    onChange={e => setFormData({...formData, username: e.target.value})}
                    required
                    disabled={!!user}
                />
            </div>
            <div>
                <label>Email:</label>
                <input
                    type="email"
                    value={formData.email}
                    onChange={e => setFormData({...formData, email: e.target.value})}
                    required
                />
            </div>
            <div>
                <label>Roles (comma separated):</label>
                <input
                    type="text"
                    value={formData.roles}
                    onChange={e => setFormData({...formData, roles: e.target.value})}
                />
            </div>
            {!user && (
                <div>
                    <label>Password:</label>
                    <input
                        type="password"
                        value={formData.password}
                        onChange={e => setFormData({...formData, password: e.target.value})}
                        required
                    />
                </div>
            )}
            <button type="submit">{user ? 'Update' : 'Create'} User</button>
        </form>
    );
};

export default UserForm; 