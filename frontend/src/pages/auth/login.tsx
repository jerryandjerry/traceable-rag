import { request } from '@/api/request'
import { getErrorMessage } from '@/api/request/error'
import { userActions } from '@/store/user'
import { LockOutlined, UserOutlined } from '@ant-design/icons'
import { Button, Card, Form, Input, message, Typography } from 'antd'
import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import './auth.scss'

const { Title, Text } = Typography

interface LoginForm {
  username: string
  password: string
}

interface LoginResponse {
  access_token: string
  token_type: 'bearer'
}

interface LoginLocationState {
  from?: {
    pathname?: string
  }
}

export function LoginPage() {
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const from =
    (location.state as LoginLocationState | null)?.from?.pathname || '/'

  const onFinish = async (values: LoginForm) => {
    setLoading(true)
    try {
      const response = await request.post<LoginResponse>('/login', values)
      const { access_token } = response.data

      userActions.setToken(access_token)
      userActions.setUsername(values.username)

      message.success('Login successful!')
      navigate(from, { replace: true })
    } catch (error: unknown) {
      message.error(getErrorMessage(error, 'Login failed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="auth-card">
      <div className="auth-header">
        <Title level={2}>Welcome Back</Title>
        <Text type="secondary">Sign in to your account</Text>
      </div>

      <Form
        name="login"
        onFinish={onFinish}
        autoComplete="off"
        layout="vertical"
        size="large"
      >
        <Form.Item
          name="username"
          rules={[
            { required: true, message: 'Please input your username!' },
            { min: 3, message: 'Username must be at least 3 characters!' },
          ]}
        >
          <Input prefix={<UserOutlined />} placeholder="Username" />
        </Form.Item>

        <Form.Item
          name="password"
          rules={[
            { required: true, message: 'Please input your password!' },
            { min: 6, message: 'Password must be at least 6 characters!' },
          ]}
        >
          <Input.Password prefix={<LockOutlined />} placeholder="Password" />
        </Form.Item>

        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>
            Sign In
          </Button>
        </Form.Item>
      </Form>

      <div className="auth-footer">
        <Text type="secondary">
          Don't have an account? <Link to="/register">Sign up</Link>
        </Text>
      </div>
    </Card>
  )
}
