import React from 'react';
import './navbar.style.scss'
import image from '../../images/mareana_logo.png'

const Navbar = () => (
    <nav className="navbar">
        <div className="container">
            <div className="logo">
                <img src={image} alt="logo" />
            </div>
            <ul className="nav">
                <a href="#">HOME</a>
            </ul>
        </div >
    </nav >
);

export default Navbar;