import { request } from '@/api/request'
import { getErrorMessage } from '@/api/request/error'
import { LockOutlined, UserOutlined } from '@ant-design/icons'
import { Button, Card, Form, Input, message, Typography } from 'antd'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import './auth.scss'

const { Title, Text } = Typography

interface RegisterForm {
  username: string
  password: string
  confirmPassword: string
}

export function RegisterPage() {
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const onFinish = async (values: RegisterForm) => {
    setLoading(true)
    try {
      await request.post('/register', {
        username: values.username,
        password: values.password,
      })

      message.success('Registration successful! Please sign in.')
      navigate('/login')
    } catch (error: unknown) {
      message.error(getErrorMessage(error, 'Registration failed'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="auth-card">
      <div className="auth-header">
        <Title level={2}>Create Account</Title>
        <Text type="secondary">Join us today</Text>
      </div>
      <Form
        name="register"
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
            {
              pattern: /^[a-zA-Z0-9_]+$/,
              message:
                'Username can only contain letters, numbers, and underscores!',
            },
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

        <Form.Item
          name="confirmPassword"
          dependencies={['password']}
          rules={[
            { required: true, message: 'Please confirm your password!' },
            ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue('password') === value) {
                  return Promise.resolve()
                }
                return Promise.reject(
                  new Error('The two passwords do not match!'),
                )
              },
            }),
          ]}
        >
          <Input.Password
            prefix={<LockOutlined />}
            placeholder="Confirm Password"
          />
        </Form.Item>

        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>
            Sign Up
          </Button>
        </Form.Item>
      </Form>

      <div className="auth-footer">
        <Text type="secondary">
          Already have an account? <Link to="/login">Sign in</Link>
        </Text>
      </div>
    </Card>
  )
}
